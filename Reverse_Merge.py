class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next  # 临时保存下一个节点
        current.next = prev       # 反转指针
        prev = current            # prev向前移动
        current = next_node       # current向前移动
    
    return prev  # 返回新的头节点

def alternate_merge(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    
    while l1 and l2:
        # 先取l1的节点
        tail.next = l1
        l1 = l1.next
        tail = tail.next
        
        # 再取l2的节点
        tail.next = l2
        l2 = l2.next
        tail = tail.next
    
    # 处理剩余节点
    tail.next = l1 if l1 else l2
    
    return dummy.next

def reverse_and_alternate_merge(l1, l2):
    # 反转两个链表
    reversed_l1 = reverse_list(l1)
    reversed_l2 = reverse_list(l2)
    
    # 交替合并
    return alternate_merge(reversed_l1, reversed_l2)

# 测试用例
def print_list(head):
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")

if __name__ == "__main__":
    # 构建测试链表
    # 链表1: 1 -> 2 -> 3
    l1 = ListNode(1)
    l1.next = ListNode(2)
    l1.next.next = ListNode(3)
    
    # 链表2: 4 -> 5 -> 6
    l2 = ListNode(4)
    l2.next = ListNode(5)
    l2.next.next = ListNode(6)
    
    print("原始链表1:")
    print_list(l1)
    print("原始链表2:")
    print_list(l2)
    
    result = reverse_and_alternate_merge(l1, l2)
    print("\n反转后交替合并的结果:")
    print_list(result)