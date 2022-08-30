summary = 'Rebecca Privitera lost 211 pounds in just over two years .\nAt her peak weight, she was 381 pounds; now she weighs 170 .\nHer husband, Justin, told her she had to lose weight the old-fashioned way .\nPrivitera has lost most of her weight by completing in-home exercise DVDs .'
data = summary.split('\n')
data = [i[:-2]+i[-1]  if i[-2]==' ' else i for i in data]
pass
