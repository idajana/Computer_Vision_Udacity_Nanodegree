import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)                     


    def forward(self, features, captions):
       
        #print(captions.shape)
        embedded_capt = self.embed(captions[:,:-1])
        #print(embedded_capt.shape)

        full_embedded= torch.cat((features.unsqueeze(1), embedded_capt), dim=1)
        #print(full_embedded.shape)
        
        lstm_out, _ = self.lstm(full_embedded)
        #print(lstm_out.shape)
        
        outputs = self.linear(lstm_out)
        #print(outputs.shape)
        
        return outputs

        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence = []
        
        while True:          
            lstm_out, states = self.lstm(inputs, states)
            
            word_vec = self.linear(lstm_out)
            
            word_vec = word_vec.squeeze(1)
            
            _, word_id = word_vec.max(dim=1)
            
               
            sentence.append(word_id.item())
            
            if word_id == 1:
                break
            inputs = self.embed(word_id)
            
            inputs = inputs.unsqueeze(1)
                    
        return sentence
            
            
            
            
            
            
            
            
        
        