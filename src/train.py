#from torch.optim import Adam

#import torch.nn.functional as F

#opt = Adam(model.parameters(), lr=1e-3)

#for epoch in range(10):
 #   model.train()
  #  for batch in train_loader:
   #     opt.zero_grad()
    #    out = model(batch)                    # [B, 2]
     #   loss = F.cross_entropy(out, batch.y.view(-1))
      #  loss.backward()
       # opt.step()

    #model.eval()
    #correct = total = 0
    #with torch.no_grad():
     #  for batch in val_loader:
      #      pred = model(batch).argmax(dim=1)
       #     correct += (pred == batch.y.view(-1)).sum().item()
        #    total   += batch.y.numel()
   # print(epoch, correct / total)
