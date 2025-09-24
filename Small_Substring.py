def smallestSubsequence(s: str) -> str:
    """返回字典序最小的子序列"""
    # 记录每个字符的最后出现位置
    last_occurrence = {c: i for i, c in enumerate(s)}
    
    stack = []
    seen = set()  # 记录栈中已有的字符
    
    for i, c in enumerate(s):
        if c in seen:
            continue  # 已经存在于栈中，跳过
        
        # 如果当前字符比栈顶字符小，且栈顶字符在后面还会出现，则弹出栈顶字符
        while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
            removed_char = stack.pop()
            seen.remove(removed_char)
        
        stack.append(c)
        seen.add(c)
    
    return ''.join(stack)

# 测试用例
print(smallestSubsequence("bcabc"))  # 输出: "abc"
print(smallestSubsequence("cbacdcbc"))  # 输出: "acdb"
print(smallestSubsequence("leetcode"))  # 输出: "letcod"