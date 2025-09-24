def suggestedProducts(products, searchWord):
    """
    优化版本：使用bisect模块
    """
    import bisect
    
    # 排序产品列表
    products.sort()
    n = len(products)
    result = []
    
    # 对于搜索词的每个前缀
    for i in range(1, len(searchWord) + 1):
        prefix = searchWord[:i]
        
        # 使用二分查找找到插入位置
        idx = bisect.bisect_left(products, prefix)
        suggestions = []
        
        # 检查最多3个可能匹配的产品
        for j in range(idx, min(idx + 3, n)):
            # 确保确实是前缀匹配（因为bisect_left可能找到的是"大于等于"的位置）
            if products[j].startswith(prefix):
                suggestions.append(products[j])
            else:
                break
        
        result.append(suggestions)
    
    return result

# 测试用例
products = ['model', 'mouse', 'music', 'moment']
searchWord = 'mouse'

print("输入:")
print(f"products = {products}")
print(f"searchWord = '{searchWord}'")
print("\n输出:")
result = suggestedProducts(products, searchWord)
for i, prefix in enumerate(searchWord):
    print(f"前缀 '{searchWord[:i+1]}': {result[i]}")