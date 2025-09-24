import heapq
from collections import defaultdict

def reorganizeString(s: str) -> str:
    # 统计字符频率
    freq = defaultdict(int)
    for char in s:
        freq[char] += 1
    
    # 检查是否有字符频率超过阈值
    max_freq = max(freq.values())
    if max_freq > (len(s) + 1) // 2:
        return ""
    
    # 构建最大堆
    heap = []
    for char, count in freq.items():
        heapq.heappush(heap, (-count, char))
    
    result = []
    prev_char = None
    
    while heap:
        # 取出当前频率最高的字符
        count1, char1 = heapq.heappop(heap)
        if char1 == prev_char:
            if not heap:
                return ""
            # 取出次高频率的字符
            count2, char2 = heapq.heappop(heap)
            result.append(char2)
            prev_char = char2
            if count2 + 1 < 0:
                heapq.heappush(heap, (count2 + 1, char2))
            heapq.heappush(heap, (count1, char1))
        else:
            result.append(char1)
            prev_char = char1
            if count1 + 1 < 0:
                heapq.heappush(heap, (count1 + 1, char1))
    
    return ''.join(result)