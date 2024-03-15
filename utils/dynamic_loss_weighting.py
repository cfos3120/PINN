import torch

class dyn_l_ws():
    def __init__(self, lr=0.001):
        
        self.W1 = torch.tensor([1],dtype=float, requires_grad=True)
        self.W2 = torch.tensor([1],dtype=float, requires_grad=True)
        self.W3 = torch.tensor([1],dtype=float, requires_grad=True)
        self.W4 = torch.tensor([1],dtype=float, requires_grad=True)

        self.params = [self.W1,self.W2,self.W3,self.W4]
        
        self.optimzer = torch.optim.Adam(self.params, lr=lr)
        self.gradloss = torch.nn.L1Loss()

        self.l01 = None
        self.l02 = None
        self.l03 = None
        self.l04 = None

        self.alpha = 0.16

    def calculate(self,model,l1,l2,l3,l4):

        # if first time, calculate reference weights
        if self.l01 == None:
            self.l01 = l1
            self.l02 = l2
            self.l03 = l3
            self.l04 = l4

        # Getting gradients of the first layers of each tower and calculate their l2-norm 
        model_param = list(model.parameters())
        G1R = torch.autograd.grad(l1, model_param[0], retain_graph=True, create_graph=True)
        G1 = torch.norm(G1R[0], 2)
        G2R = torch.autograd.grad(l2, model_param[0], retain_graph=True, create_graph=True)
        G2 = torch.norm(G2R[0], 2)
        G3R = torch.autograd.grad(l3, model_param[0], retain_graph=True, create_graph=True)
        G3 = torch.norm(G3R[0], 2)
        G4R = torch.autograd.grad(l4, model_param[0], retain_graph=True, create_graph=True)
        G4 = torch.norm(G4R[0], 2)
        G_avg = (G1 + G2 + G3 +G4)/4

        # Calculating relative losses 
        lhat1 = l1/self.l01
        lhat2 = l2/self.l02
        lhat3 = l3/self.l03
        lhat4 = l4/self.l04
        lhat_avg = (lhat1 + lhat2 + lhat3 + lhat4)/4

        # Calculating relative inverse training rates for tasks 
        inv_rate1 = lhat1/lhat_avg
        inv_rate2 = lhat2/lhat_avg
        inv_rate3 = lhat3/lhat_avg
        inv_rate4 = lhat4/lhat_avg

        # Calculating the constant target for Eq. 2 in the GradNorm paper
        C1 = (G_avg*(inv_rate1)**self.alpha).detach()
        C2 = (G_avg*(inv_rate2)**self.alpha).detach()
        C3 = (G_avg*(inv_rate3)**self.alpha).detach()
        C4 = (G_avg*(inv_rate4)**self.alpha).detach()

        self.optimzer.zero_grad()

        # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
        self.Lgrad = self.gradloss(G1, C1) + self.gradloss(G2, C2) + self.gradloss(G3, C3) + self.gradloss(G4, C4)
        self.Lgrad.backward(retain_graph=True)

        # Updating loss weights 
        self.optimzer.step()

    def renorm(self):
        # Renormalizing the losses weights
        coef = 4 / (self.W1 + self.W2 + self.W3 + self.W4)
        self.params = [coef*self.W1, coef*self.W2, coef*self.W3, coef*self.W4]