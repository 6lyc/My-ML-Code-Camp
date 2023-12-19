import torch
import torch.nn.functional as F

def evaluate(best_val_acc, best_test_acc, model, g, features, labels, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute accuracy on training/validation/test
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        # 用验证集的最佳acc定模型参数
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return val_acc, test_acc, best_val_acc, best_test_acc


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    best_val_acc = 0
    best_test_acc = 0
    for e in range(200):
        # Forward
        model.train()
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc, test_acc, best_val_acc, best_test_acc = evaluate(best_val_acc, best_test_acc, model, g, features, labels, val_mask, test_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))