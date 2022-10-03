from sqlalchemy import false
from torch import nn
import argparse
from typing import Any, Dict
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from module.utils import *
from module.prefix_Bart import BartForConditionalGeneration,PretrainedBartModel
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}
class PrefixTransformer(nn.Module):
    def __init__(
        self,
        hparams,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        seq2seq_model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.hparams = hparams
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading
        
        cache_dir = hparams["cache_dir"] if "cache_dir" in hparams.keys() else None
        print('the cache dir is {}'.format(cache_dir))
        if config is None:
            self.config = AutoConfig.from_pretrained(
                hparams["config_name"] if hparams["config_name"] else hparams["model_name_or_path"],
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if p in hparams.keys():
                assert hparams[p], f"model config doesn't have a `{p}` attribute"


        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                hparams["config_name"] if hparams["config_name"] else hparams["model_name_or_path"],
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        # transformer版本差距进行的补充参数
        self.config.normalize_embedding = True
        self.config.static_position_embeddings = False
        self.config.extra_pos_embeddings =2
        # print(self.hparams.preseqlen)
        self.config.preseqlen = hparams["preseqlen"]
        self.config.use_prefix = True

        self.seq2seq_model_type = MODEL_MODES[mode]
        if seq2seq_model is None:
            self.seq2seq_model = BartForConditionalGeneration.from_pretrained(
                hparams["model_name_or_path"],
                from_tf=bool(".ckpt" in hparams["model_name_or_path"]),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.seq2seq_model = seq2seq_model



        config_prefix = AutoConfig.from_pretrained(hparams["model_name_or_path"], cache_dir=cache_dir)
        self.model_type = config_prefix.model_type

        if hparams["optim_prefix"] == 'yes':
            optim_prefix_bool = True
        elif hparams["optim_prefix"] == 'no':
            optim_prefix_bool = False
        else:
            assert False, "model_args.optim_prefix should be either yes or no"

        print(self.model_type)
        config_prefix._my_arg_tune_mode = hparams["tuning_mode"]
        config_prefix._my_arg_task_mode = hparams["task_mode"]
        config_prefix._my_arg_control = True
        config_prefix.train_weights = False
        config_prefix.optim_prefix = optim_prefix_bool
        config_prefix.use_deep = hparams["use_deep"] 
        config_prefix.preseqlen = hparams["preseqlen"]
        config_prefix.use_infix = (hparams["format_mode"] == 'infix')
        config_prefix.format_mode = hparams["format_mode"]
        config_prefix.prefix_dropout =hparams["prefix_dropout"]
        config_prefix.vocab_size = len(self.tokenizer)
        # some extra stuff.
        config_prefix.mid_dim = hparams["mid_dim"]

        # print(config_prefix)

        if hparams["prefixModel_name_or_path"]:
            print('loading from {}'.format(hparams["prefixModel_name_or_path"]))
            self.model = PrefixTuning.from_pretrained(hparams["prefixModel_name_or_path"],
                        cache_dir=cache_dir,
                        config=config_prefix,
                        model_gpt2=self.seq2seq_model)
        else:
            self.model = PrefixTuning(config_prefix, self.seq2seq_model)



    def load_hf_checkpoint(self, *args, **kwargs):
        assert False, 'why need to load model here?'
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, mode):
        if mode == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def save_checkpoint(self, trainer) -> None:
        print('Saving the the checkpoint.')
        return
    def on_save_checkpoint(self, checkpoint: Dict[str, Any], filepath=None) -> None:
        # if filepath is not None:
        #     save_path = filepath[:-5]
        # else:
        #     save_path = self.output_dir.joinpath("checkpoint-hello")
        save_path = filepath #self.output_dir.joinpath("checkpoint-curr_best")
        print('the suggested save_path is {}, saving to {}'.format(filepath, save_path))

        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print('SAVING TO checkpoint {}'.format(save_path))

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        parser.add_argument(
            "--prefixModel_name_or_path",
            default=None,
            type=str,
            help="Path to pretrained prefix model or model identifier from huggingface.co/models",
        )

        parser.add_argument(
            "--prefix_mode",
            default='activation',
            type=str,
            help="embedding or activation",
        )

        parser.add_argument(
            "--preseqlen",
            default=1,
            type=int,
            help="the length of the prefix.",
        )

        parser.add_argument(
            "--use_deep",
            default='no',
            type=str,
            help="use the deep optimization of the prefix.",
        )

        parser.add_argument(
            "--optim_prefix",
            default='yes',
            type=str,
            help="use the task specific optimization of the prefix.",
        )

        parser.add_argument(
            "--tuning_mode",
            default='prefixtune',
            type=str,
            help="Could be prefixtune or finetune",
        )

        parser.add_argument(
            "--prefix_dropout",
            default=0.0,
            type=float,
            help="the dropout rate for our prefix model.",
        )

        parser.add_argument(
            "--use_dropout",
            default='no',
            type=str,
            help="whether to dropout the main model during training. ",
        )

        parser.add_argument(
            "--mid_dim",
            default=512,
            type=int,
            help="the dimension of the intermediate layer.",
        )

        # parser.add_argument(
        #     "--task_mode",
        #     default='summarization',
        #     type=int,
        #     help="the default task, or dataset name. ",
        # )

        parser.add_argument(
            "--format_mode",
            default='cat',
            type=str,
            help="whether to look at the input again, including [infix, cat, peek, nopeek]",
        )


        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default=None,
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=10, type=int)
        parser.add_argument("--eval_batch_size", default=10, type=int)
        parser.add_argument("--adafactor", action="store_true")
class PrefixSummarizationModule(PrefixTransformer):
    mode = "summarization"


    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        
        self.step_count = 0
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        # if self.hparams.freeze_embeds:
        #     self.freeze_embeds()

        freeze_params(self.seq2seq_model) # 冻结参数
        assert_all_frozen(self.seq2seq_model)
        print('FREEZING ENTIRE seq2seq model.')
        # if self.hparams.freeze_encoder:
        #     freeze_params(self.model.get_encoder())
        #     assert_all_frozen(self.model.get_encoder())
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.eval_max_length = 62
        self.eval_min_length = 11
        self.eval_beams =6
        print('for deocding, eval_max_length={}, '
              'eval_min_length={}, eval_beams={}'.format(self.eval_max_length, self.eval_min_length, self.eval_beams))

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        if self.model_type == "t5":
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)
        elif self.model_type == "fsmt":
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        else:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, gpt2_model=self.seq2seq_model, **kwargs)
    

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

class PrefixTuning(PretrainedBartModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, optim_prefix=False, preseqlen=5, use_infix=False, deep_param=False):
        super().__init__(config)
        print('under the PrefixTuning model')# 使用prefix

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head



        if hasattr(config, 'optim_prefix'):
            self.optim_prefix = config.optim_prefix
        else:
            self.optim_prefix = optim_prefix

        if hasattr(config, 'preseqlen') and self.optim_prefix:
            self.preseqlen = config.preseqlen
        elif self.optim_prefix:
            self.preseqlen = preseqlen

        if hasattr(config, 'use_infix'):
            self.use_infix = config.use_infix
        else:
            self.use_infix = use_infix

        if hasattr(config, 'use_deep'):
            self.use_deep = (config.use_deep == 'yes')
        else:
            self.use_deep = False

        deep_param = self.use_deep

        
        if hasattr(config, '_my_arg_tune_mode'):
            self.tuning_mode = config._my_arg_tune_mode
        else:
            self.tuning_mode = 'prefixtune'

        if hasattr(config, '_my_arg_task_mode'):
            self.task_mode = config._my_arg_task_mode
        else:
            self.task_mode = 'underspecified'
            assert False, 'the task is underspecified'

        if hasattr(config, 'train_weights'):
            self.train_weights = (config.train_weights == 'yes')
        else:
            assert False, "unspecified train weights"

        if hasattr(config, 'format_mode'):
            self.format_mode = config.format_mode
        else:
            self.format_mode = 'cat'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        # config_prefix.init_random = model_args.init_random
        # config_prefix.mid_dim = model_args.mid_dim

        if hasattr(config, 'init_random'):
            self.init_random = (config.init_random == 'yes')
        else:
            self.init_random = False

        if hasattr(config, 'mid_dim'):
            self.mid_dim = config.mid_dim
        else:
            self.mid_dim = 512

        if hasattr(config, 'lowdata'):
            self.lowdata = config.lowdata
        else:
            self.lowdata = False

        if hasattr(config, 'lowdata_token'):
            self.lowdata_token = config.lowdata_token
        else:
            self.lowdata_token = None


        if self.task_mode == 'dataless':
            self.mode_para = 1
        elif self.task_mode == 'data2text' or self.task_mode == 'triples' or self.task_mode == 'webnlg' or \
                self.task_mode == 'writingPrompts':
            # with src and input based encoding.
            self.mode_para = 2
            # self.mode_para=0 and optim_prefix == True for Instruction based.
        else:
            self.mode_para = 4

        if not self.optim_prefix:
            if self.train_weights:
                self.wte = model_gpt2.transformer.wte
                for p in self.wte.parameters():
                    p.requires_grad = True
            else:
                if not self.init_random:
                    self.wte = None
                else:
                    print('the is just for baseline checking!!! We reinitialize the LM embeddings and try cat '
                          'and peek.')
                    print('BASELINE'*100)
                    self.wte = nn.Embedding(config.vocab_size, config.n_embd)
                    print(self.wte)



            if self.mode_para == 1:
                print('mode_para=1, for dataless.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p4_infix
                else:
                    self.get_prompt = self.get_prompt_p4
            elif self.mode_para == 2 or self.mode_para == 4:
                print('mode_para=2 or 4, for (2)data2text having a variable length input prefix parametrization. or for (4) topic/keyword/attributes...')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p3_infix
                else:
                    self.get_prompt = self.get_prompt_p3


            elif self.mode_para == 3:
                print('mode_para=3, OLD VERSION: many parameters.')
                self.control_trans = nn.Sequential(
                    nn.Linear(config.n_embd, self.preseqlen * config.n_layer * 2 * config.n_embd), nn.Tanh())
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p1_infix
                else:
                    self.get_prompt = self.get_prompt_p1
        else:
            self.mode_para = 0
            print('mode_para=0, for data2text Instruction based, just optimize a set of parameters ;) ')
            print('preseqlen is {}, under the mode of optimizing prefix directly'.format(self.preseqlen))


            if self.lowdata and self.lowdata_token is not None:
                low_data_init = 3
                if low_data_init == 1:
                    print('IN THE LOW DATA SETTING, EXPLORE INITIALIZATION FOR DIRECT OPTIM...')
                    # self.control_trans = nn.Parameter(torch.randn(self.preseqlen * config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p22
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)
                    self.control_trans = self.lowdata_init_train1(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
                    print(self.control_trans.shape)
                elif low_data_init == 2:
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, need to train first')
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5

                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
                    # sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    sample_text = 'name : Blue Spice | Type : coffee shop | customer rating : 5 out of 5 | near : Crowne Plaza Hotel||The coffee shop Blue Spice is based near Crowne Plaza Hotel and has a high customer rating of 5 out of 5 .'
                    src, tgt = sample_text.split('||')
                    sample_input = ' {} {} '.format(src, tokenizer.bos_token) + tgt + ' {}'.format(tokenizer.eos_token)

                elif low_data_init == 3:
                    # use a single prepended token.
                    assert self.lowdata_token is not None
                    self.preseqlen = len(self.lowdata_token[0])
                    print('IN THE LOW DATA SETTING, UNDER PARAMETRIZATION 1, low_data_init=3, '
                          'preseqlen = {} Unifying with FINETUNE'.format(self.preseqlen))
                    self.input_tokens = torch.arange(self.preseqlen).long()
                    self.wte = nn.Embedding(self.preseqlen, config.n_embd)
                    self.control_trans = nn.Sequential(
                        nn.Linear(config.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
                    self.get_prompt = self.get_prompt_p5






            # DIFFERENT PARAMETRIZATION:
            elif not deep_param:
                low_data_init = 0
                print('UNDER PARAMETRIZATION 1')
                self.input_tokens = torch.arange(self.preseqlen).long()# [0,1,2,3..............200]
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)# 200*1024
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))# 输入维度1024  输出维度24576
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))



            else:
                low_data_init = 0
                print('UNDER PARAMETRIZATION DEEP 1')

                self.input_tokens = torch.arange(self.preseqlen).long()
                self.wte = nn.Embedding(self.preseqlen, self.n_embd)# 200*1024
                self.control_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))
                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5


                if self.use_infix:
                    self.get_prompt = self.get_prompt_p5_infix
                else:
                    self.get_prompt = self.get_prompt_p5

                self.use_encoder_prefix = True
                self.use_cross_prefix = True

                if self.use_encoder_prefix:
                    self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans_enc = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

                if self.use_cross_prefix:
                    self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
                    self.control_trans2 = nn.Sequential(
                        nn.Linear(self.n_embd, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.mid_dim),
                        nn.Tanh(),
                        nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))


        self.dropout = nn.Dropout(self.prefix_dropout)# 不进行dropout
        if self.use_infix:
            self.forward = self.forward_infix

        ###### just trying #########
        total_param = 0
        for name, param in self.named_parameters():
            print("name is {},papram shpe is {}".format(name,param.shape))
            total_param += param.numel()
        print('total param is {}'.format(total_param))

        if low_data_init == 2:
            self.lowdata_init_train2(gpt2=model_gpt2, tokenizer=tokenizer, sample_input=sample_input)
        elif low_data_init == 3:
            print('use pt for this tensor', torch.LongTensor(self.lowdata_token))
            self.lowdata_init_train3(gpt2=model_gpt2, sample_input=torch.LongTensor(self.lowdata_token))



    def lowdata_init_train1(self, gpt2, tokenizer, sample_input):
        input = tokenizer(sample_input, return_tensors='pt')
        output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
        output = output.past_key_values
        print(len(output), output[0].shape)
        output = torch.cat(output, dim=0).detach()
        return torch.nn.Parameter(output)

    def get_prompt_p22(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        past_key_values = self.control_trans.expand(-1, bsz, -1, -1, -1).split(2, dim=0)
        return past_key_values

    def lowdata_init_train2(self, gpt2, tokenizer, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            input = tokenizer(sample_input, return_tensors='pt')
            output = gpt2(input['input_ids'].to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()

        return


    def lowdata_init_train3(self, gpt2, sample_input, epochs=500): # prev=500
        self = self.cuda()
        gpt2 = gpt2.cuda()
        with torch.no_grad():
            output = gpt2(sample_input.to(gpt2.device), return_dict=True, use_cache=True)
            output = output.past_key_values
            print(len(output), output[0].shape)
            output = torch.cat(output, dim=0)

        optimizer_temp = torch.optim.Adam(self.control_trans.parameters(), lr=0.0001)

        for e in range(epochs):
            our_prompt = self.get_prompt_p5(bsz=1)
            our_prompt = torch.cat(our_prompt, dim=0)
            loss_metrics = nn.MSELoss()
            loss = loss_metrics(our_prompt.to(gpt2.device), output)
            print(loss)
            loss.backward()
            optimizer_temp.step()
            self.control_trans.zero_grad()
        return

    def get_prompt_p2(self, control_code=None, gpt2=None, bsz=None):
        assert bsz is not None
        temp_control = self.control_trans.view(1, self.preseqlen,  self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd).expand(bsz, -1, -1, -1, -1)
        temp_control = self.dropout(temp_control)
        past_key_values = temp_control.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p3_infix(self, src, control_code=None, gpt2=None, bsz=None):
        # temp_result = gpt2(inputs_embeds=input_embs, use_cache=True, return_dict=True)
        # print('infix')
        src_out = gpt2(input_ids=src, use_cache=True, return_dict=True, output_hidden_states=True)
        src_repr = src_out.hidden_states[-1] #bsz, seqlen, hidden
        src_past_key_vals = src_out.past_key_values
        past_key_values = self.control_trans(src_repr) #bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        # print(past_key_values.shape)
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        full_lst = []
        for i in range(len(src_past_key_vals)):
            full_lst.append(torch.cat([src_past_key_vals[i], past_key_values[i]], dim=3))

        return full_lst

    def get_prompt_p3(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
        return past_key_values


    def get_prompt_p5(self, control_code=None, gpt2=None, bsz=None, sample_size=1):
        # 注释内容是训练过程中用的 self.input_tokens 200
        old_bsz = bsz
        bsz = bsz * sample_size# 16*1
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)# self.input_tokens 200 ； input_tokens [16*200]
        temp_control = self.wte(input_tokens)# 16*200*1024
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb 16*200*24576  24*1024
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)# 16*200*24*16*64
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # list:12 2*16*16*200*64



        if self.use_cross_prefix:
            temp_control2 = self.wte2(input_tokens)# 16*200*1024
            past_key_values2 = self.control_trans2(temp_control2)   #bsz, seqlen, layer*emb 16*200*24576  24*1024
            bsz, seqlen, _ = past_key_values2.shape
            past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values2 = self.dropout(past_key_values2)
            past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)# list:12 2*16*16*200*64


        if self.use_encoder_prefix:
            input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
            temp_control_enc = self.wte_enc(input_tokens_enc)
            past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                     self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                  "prev_value": key_val[1].contiguous(),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                 },
                        }
            if self.use_cross_prefix:
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                                }
            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                        }
            result.append(temp_dict)
# result {self     encoder_decoder encoder} prev_key prev_value prev_key_padding_mask
        return result

    def get_prompt_p6(self, control_code=None, gpt2=None, bsz=None):
        input_embs = self.input_embs.to(self.device)
        past_key_values = self.control_trans(input_embs).expand(bsz, -1, -1) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def get_prompt_p4(self, control_code, gpt2=None, bsz=None):
        # print(control_code, control_code.shape)
        if control_code is not None:
            if self.wte:
                temp_control = self.wte(control_code)
            else:
                assert gpt2 is not None
                temp_control = gpt2.transformer.wte(control_code) #bsz, seqlen, emb
            # need to handle padding? use attention mask.
            # print(temp_control.shape)
            past_key_values = self.control_trans(temp_control).mean(1).unsqueeze(1) #bsz, seqlen, layer*emb
            bsz, seqlen, _ = past_key_values.shape
            # print(past_key_values.shape)
            past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def get_prompt_p1(self, control_code, gpt2=None, bsz=None):
        if control_code is not None:

            if type(control_code) is tuple :
                assert False, 'Tuples'
                control_embs, control_word = control_code
                past_key_values = self.control_trans(control_embs)
                past_key_values = past_key_values.mean(1).unsqueeze(1)
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen * self.preseqlen, self.match_n_layer * 2,
                                                       self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                print(control_word, control_embs.shape)
            else:
                # print('running with control code')
                # use the control code to generate the first 5 activation layers.
                if not self.embMatch:
                    if self.wte:
                        temp_control = self.wte(control_code)
                    else:
                        assert gpt2 is not None
                        temp_control = gpt2.transformer.wte(control_code)
                    temp_control = temp_control.sum(1).unsqueeze(1)
                else:
                    temp_control = control_code
                    # print(control_code.shape)
                past_key_values = self.control_trans(temp_control)
                # print(past_key_values.shape) #bsz, controlCodeLen, long... 5 * config.n_layer * 2 * config.n_embd
                past_key_values = past_key_values.sum(1).unsqueeze(1)
                # print(past_key_values.shape)  # bsz, 1, long...
                bsz, seq_pastlen, _ = past_key_values.shape
                past_key_values = past_key_values.view(bsz, seq_pastlen*self.preseqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        else:
            assert False, "control_code is None"
            past_key_values = None
        return past_key_values

    def forward(self,
        input_ids=None,
        gpt2_model=None,
        past_key_values=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # labels=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # if self.mode_para == 2:
        #     past_key_values_prompt = self.get_prompt(src, gpt2=gpt2_model, bsz=bsz)
        # else:

        past_key_values_prompt = self.get_prompt(bsz=bsz)

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)


        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, **kwargs)

        # output = gpt2_model(input_ids=input_ids,
        #                     past_key_values=past_key_values, attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids, position_ids=position_ids,
        #                    head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        #                    encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
        #                    output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        #                    return_dict=return_dict, **kwargs)

        return output# output[0] bsz*seq_len*50264 output[1] bsz*1024*1024


    def forward_infix(self,
        input_ids=None,
        weights=None,
        control_code=None,
        emb_match=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        cate_batch=None,
        cate_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        if self.mode_para == 2:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1) # bsz, seqlen
        else:
            past_key_values_prompt = self.get_prompt(src, None, gpt2=gpt2_model, bsz=bsz)
            attention_mask = torch.cat([src_attn, src_attn, tgt_attn], dim=1)  # bsz, seqlen

        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"


        output = gpt2_model(input_ids=input_ids, control_code=None, weights=weights, emb_match=emb_match,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output


