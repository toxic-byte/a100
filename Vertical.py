from collections import deque, defaultdict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def verticalTraversal(root):
    if not root:
        return []
    
    # 使用字典存储列索引对应的节点值
    column_table = defaultdict(list)
    # 队列存储节点和对应的列索引
    queue = deque([(root, 0)])
    
    while queue:
        level_size = len(queue)
        # 临时存储当前层的节点
        level_nodes = defaultdict(list)
        
        for _ in range(level_size):
            node, col = queue.popleft()
            level_nodes[col].append(node.val)
            
            if node.left:
                queue.append((node.left, col - 1))
            if node.right:
                queue.append((node.right, col + 1))
        
        # 对当前层的节点按列排序后存入结果
        for col in level_nodes:
            column_table[col].extend(sorted(level_nodes[col]))
    
    # 按列索引排序后输出结果
    return [column_table[col] for col in sorted(column_table)]

# 测试用例
if __name__ == "__main__":
    # 构建测试树
    #       3
    #      / \
    #     9  20
    #       /  \
    #      15   7
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    
    print(verticalTraversal(root))  # 输出: [[9], [3, 15], [20], [7]]