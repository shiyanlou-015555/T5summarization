# int 数字 0-999  0-zero 
uninum = ["zero","one","two","three","four","five","six","seven","night","nine","ten","eleven","tweleve","threeteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
twonum = ["","ten","twity","thirty","fourty","fifty","sixty","seventy","nighty","ninety"]
def res_out(i):
    if i < 20:
        return uninum[i]
    if i<100:
        if i%10!=0:
            return twonum[i//10] +" "+uninum[i%10]
        else:
            return twonum[i//10]
    if i%100==0:
        res = uninum[i//100]+ " hundred"
    else:
        res = uninum[i//100]+ " hundred and "+res_out(i%100)
    return res
print(res_out(999))

