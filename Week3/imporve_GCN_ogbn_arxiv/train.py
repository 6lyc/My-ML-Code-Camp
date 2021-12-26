from getdata import getdata
from model import StochasticTwoLayerGCN
import torch
import torch.nn.functional as F
import tqdm
import sklearn.metrics
import numpy as np

g, train_dataloader, valid_dataloader, test_dataloader, node_labels = getdata()
node_features = g.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StochasticTwoLayerGCN(num_features, round(num_features/2), num_classes).to(device)

opt = torch.optim.Adam(model.parameters())
best_accuracy = 0
best_model_path = 'model.pt'
for epoch in range(500):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

    model.eval()

    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

best_model = StochasticTwoLayerGCN(num_features, round(num_features/2), num_classes).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
predictions = []
labels = []
with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
    for input_nodes, output_nodes, mfgs in tq:
        inputs = mfgs[0].srcdata['feat']
        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    print('Test Accuracy {}'.format(accuracy))
