import torch
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import SimpleITK as sitk
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size =3, stride =1, padding =1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
            )
        self.relu = nn.ReLU(inplace = True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        #print(g1.shape)
        #print(x1.shape)
        g1 = F.interpolate(g1, scale_factor = (x1.shape[2]/g1.shape[2],x1.shape[3]/g1.shape[3],x1.shape[4]/g1.shape[4]), mode = 'trilinear')
        #print(g1.shape)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        #print(psi.shape)

        return x*psi

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_in, ch_out, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor =2),
            nn.Conv3d(ch_in,ch_out,3,stride =1 ,padding =1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inpalce =True)
        )
    def forward(self, x):
        x =self.up(x)
        return x    



class Att_Unet(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, training = True):
        super(Att_Unet, self).__init__()
        self.trianing = training
        # encoder section
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.Conv3d(32, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True)
        )

        self.att1 = Attention_block(F_g =128, F_l =128, F_int =128)

        self.encoder5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace = True)
        )

        # decoder section
        self.decoder1 = nn.Sequential(
            nn.Conv3d(384, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        )
        self.att2 = Attention_block(F_g=64, F_l=64, F_int =64)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.att3 = Attention_block(F_g =32, F_l=32, F_int =32)
        self.decoder4 = nn.Sequential(
            nn.Conv3d(32, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv3d(64, 32, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv3d(32, 2, 3, stride = 1, padding = 1),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace = True)
        )
        self.decoder7 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1)
        )


        
    def forward(self, x):
        # encoder section
        x = x
        x = self.encoder1(x)# relu(4->8)
        f1 = x #(8)
        f1= f1
        #print("encoder1_size:",f1.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(8->8)
        #print("test",x.shape)
        x = self.encoder2(x)# relu(8->16)
        f2 = x #(16)
        f2 = f2
        #print("encoder2_size:",f2.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(16->16)
        
        x = self.encoder3(x)# relu(16->32)
        f3 = x #(32)
        f3 =f3
        #print("encoder3_size:",f3.shape)
        x = F.max_pool3d(x,kernel_size = 2,stride = 2,padding = 0)# maxpool(32->32)
        x = self.encoder4(x)# relu(32->32)
        a1 = x
        x=x
        a1=a1
        
        A1 = self.att1(g = f3, x =a1)#attention block1
        x = self.encoder5(x)#relu(32->64)
        
        
        # decoder section

        x = F.interpolate(x, scale_factor = (f3.shape[2]/x.shape[2],f3.shape[3]/x.shape[3],f3.shape[4]/x.shape[4]), mode = 'trilinear')# upsample（16->16)
        
        A1 = F.interpolate(A1, scale_factor = (f3.shape[2]/A1.shape[2],f3.shape[3]/A1.shape[3],f3.shape[4]/A1.shape[4]), mode = 'trilinear')# upsample（16->16)

        x = torch.cat((x,A1),1) #(64+32 = 96)
        
        x = x
        x = self.decoder1(x) # relu(96 ->32)
        

        a2 = x #attention block2, size = 32
        a2 = a2

        f2 = f2
        A2 = self.att2(g =f2,x=a2) # size =32

        A2 = F.interpolate(A2, scale_factor = (f2.shape[2]/A2.shape[2],f2.shape[3]/A2.shape[3],f2.shape[4]/A2.shape[4]), mode = 'trilinear') # upsample（16->16)
      #   x = x
        #print("decoder1_size:",x.shape)
        x = self.decoder2(x) #relu(32->32)
        
        x= x
        x = F.interpolate(x, scale_factor =(f2.shape[2]/x.shape[2],f2.shape[3]/x.shape[3],f2.shape[4]/x.shape[4]), mode = 'trilinear') # upsample(256->256)

        x = torch.cat((x,A2),1) #(4+4 = 8) 
        x = x
        
        x = self.decoder3(x) # relu(8 ->2)
        
        #print("decoder2_size:",x.shape)
        a3 = x#attention block2

        f1 = f1
        a3  = a3
        A3 = self.att3(g =f1,x=a3)
        
        A3 = F.interpolate(A3, scale_factor = (f1.shape[2]/A3.shape[2],f1.shape[3]/A3.shape[3],f1.shape[4]/A3.shape[4]), mode = 'trilinear') # upsample（16->16)

        x = x
        x = self.decoder4(x) #relu(2->2)
        
        x = F.interpolate(x, scale_factor =(f1.shape[2]/x.shape[2],f1.shape[2]/x.shape[2],f1.shape[2]/x.shape[2]), mode = 'trilinear') # upsample(128->128)
        
        x = torch.cat((x,A3),1) #(2+2 = 4)
        
        # x= x.to("cuda:2")
        x = self.decoder5(x) # relu(4 ->2)
        
        #print("decoder3_size:",x.shape)
        x = self.decoder6(x)
        
        x = self.decoder7(x)
        
        x = F.interpolate(x, scale_factor = (f1.shape[2]/x.shape[2],f1.shape[3]/x.shape[3],f1.shape[4]/x.shape[4]), mode = 'trilinear')

        x = F.sigmoid(x)

        return x
    
def main():
    x = torch.randn(1,1,56,56,56)
    print(x.shape)
    model = Att_Unet()
    y = model(x)
    print(y.shape)
    import pdb
    pdb.set_trace()



if __name__ == "__main__":
    main()