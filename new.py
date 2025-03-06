import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class lmodel(nn.Module):
    def __init__(self):
        super(lmodel,self).__init__()
        self.l1 = nn.Linear(7,126)
        self.l2 = nn.Linear(126,126)
        self.l3 = nn.Linear(126,64)
        self.l4 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,x):
        x = self.l1(x) #input layer
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.relu(x)

        x = self.l4(x) #output layer
        x = self.sigmoid(x)
        
        return x
    
model = torch.load("model.pth", weights_only=False)

#validation
columns = ["GENERAL APPEARANCE", "MANNER OF SPEAKING", "PHYSICAL CONDITION", 
           "MENTAL ALERTNESS", "SELF-CONFIDENCE", "ABILITY TO PRESENT IDEAS", 
           "COMMUNICATION SKILLS"]

value=[]
ans = ""
for v in columns:
    ans = int(input("your " + v + " (rate 1-5, integer only.)"))
    while(ans <0 or ans >5):
        ans = int(input("your " + v + " (rate 1-5, integer only.)"))
        
    value.append(ans)

value = np.array(value)
value = value.reshape(1,7)
print(value)

value = scaler.fit_transform(value)
inTensor=torch.tensor(value,dtype=torch.float32)

with torch.no_grad():
    pred = model.forward(inTensor)

print(f"chances of getting employed: {round(float(pred)*100,2)}%")