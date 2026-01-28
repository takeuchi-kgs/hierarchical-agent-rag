function toggleNode(childrenId, iconId) {
    const childrenElement = document.getElementById(childrenId);
    const iconElement = document.getElementById(iconId);
    
    if (!childrenElement || !iconElement) return;
    
    if (childrenElement.classList.contains('collapsed')) {
        childrenElement.classList.remove('collapsed');
        iconElement.textContent = '▼';
    } else {
        childrenElement.classList.add('collapsed');
        iconElement.textContent = '▶';
    }
}

// 初期状態: ルートと最初のレベルだけを展開した状態に設定
document.addEventListener('DOMContentLoaded', function() {
    // すべての2番目のレベル以下の子ノードを折りたたむ
    const allChildren = document.querySelectorAll('.node-children');
    allChildren.forEach(function(children) {
        // ルートの直接の子でない場合は折りたたむ
        const parentNode = children.closest('.tree-node');
        if (parentNode && !parentNode.classList.contains('video-node')) {
            const grandParent = parentNode.parentElement;
            if (grandParent && !grandParent.classList.contains('tree-container')) {
                children.classList.add('collapsed');
                const iconId = children.id.replace('-children', '-icon');
                const icon = document.getElementById(iconId);
                if (icon) {
                    icon.textContent = '▶';
                }
            }
        }
    });
});
