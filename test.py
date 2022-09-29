def get_substr(nums1,nums2):
    result = 0
    res = [0 for idx in range(len(nums2)+1)]
    temp_j = 0
    for i in range(1,len(nums1)+1):
        for j in range(len(nums2),0,-1):
            if nums1[i-1]==nums2[j-1]:
                res[j] = res[j-1]+1
            else:
                res[j]==0
            if result < res[j]:
                result=res[j]
                temp_j = j
    return nums2[temp_j-result:temp_j]
nums1 = "baidu"
nums2 = "xaidx"
print(get_substr(nums1,nums2))