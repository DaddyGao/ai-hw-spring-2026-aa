import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. Model Definition
# ==========================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. Attack Algorithms
# ==========================================
def fgsm_attack(image, epsilon, data_grad):
    """Fast Gradient Sign Method"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Ensure pixels remain in valid range [0, 1]
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, image, label, epsilon, alpha, iters):
    """Projected Gradient Descent / Iterative FGSM"""
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    
    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, label)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        # Step in the direction of the gradient
        perturbed_image = perturbed_image.detach() + alpha * data_grad.sign()
        
        # Project back to the epsilon-ball around original image
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0, max=1).detach()
        
    return perturbed_image

def mifgsm_attack(model, image, label, epsilon, alpha, iters, decay=1.0):
    """Momentum Iterative FGSM"""
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    momentum = torch.zeros_like(image).detach()
    
    for _ in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(output, label)
        
        model.zero_grad()
        loss.backward()
        
        grad = perturbed_image.grad.data
        
        # Normalize gradient and update momentum
        grad_norm = torch.norm(grad, p=1)
        grad = grad / (grad_norm + 1e-8)
        momentum = decay * momentum + grad
        
        # Step in the direction of momentum
        perturbed_image = perturbed_image.detach() + alpha * momentum.sign()
        
        # Project back to the epsilon-ball
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0, max=1).detach()
        
    return perturbed_image

# ==========================================
# 3. Training Loop
# ==========================================
def train(model, device, train_loader, optimizer, epochs=3):
    model.train()
    print("--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 300 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss.item():.6f}')

# ==========================================
# 4. Evaluation & Attack Loop
# ==========================================
def test_attacks(model, device, test_loader, epsilon):
    model.eval()
    
    correct_clean = 0
    total_samples = 0
    
    # Trackers for ASR
    fgsm_success = 0
    pgd_success = 0
    mifgsm_success = 0
    
    print(f"\n--- Running Attacks (Epsilon = {epsilon}) ---")
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        total_samples += len(data)
        
        # 1. Baseline Clean Prediction
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        
        # Create a mask of originally correct predictions
        correct_mask = (init_pred.flatten() == target).flatten()
        correct_clean += correct_mask.sum().item()
        
        # If the model got everything wrong in this batch, skip attacks
        if correct_mask.sum() == 0:
            continue
            
        # Keep ONLY images the model originally classified correctly
        data_c = data[correct_mask]
        target_c = target[correct_mask]
        
        # --- FGSM ---
        data_c.requires_grad = True
        output_c = model(data_c)
        loss = F.nll_loss(output_c, target_c)
        model.zero_grad()
        loss.backward()
        data_grad = data_c.grad.data
        
        adv_fgsm = fgsm_attack(data_c, epsilon, data_grad)
        pred_fgsm = model(adv_fgsm).max(1, keepdim=True)[1].flatten()
        fgsm_success += (pred_fgsm != target_c).sum().item()
        
        # --- PGD / I-FGSM ---
        adv_pgd = pgd_attack(model, data_c, target_c, epsilon, alpha=0.01, iters=40)
        pred_pgd = model(adv_pgd).max(1, keepdim=True)[1].flatten()
        pgd_success += (pred_pgd != target_c).sum().item()
        
        # --- MI-FGSM ---
        adv_mifgsm = mifgsm_attack(model, data_c, target_c, epsilon, alpha=0.01, iters=40, decay=1.0)
        pred_mifgsm = model(adv_mifgsm).max(1, keepdim=True)[1].flatten()
        mifgsm_success += (pred_mifgsm != target_c).sum().item()

    # Calculate Metrics
    clean_acc = correct_clean / total_samples
    asr_fgsm = fgsm_success / correct_clean
    asr_pgd = pgd_success / correct_clean
    asr_mifgsm = mifgsm_success / correct_clean
    
    print("\n--- Final Metrics ---")
    print(f"Recognition Rate (Clean Accuracy): {clean_acc*100:.2f}%")
    print(f"FGSM ASR:       {asr_fgsm*100:.2f}%")
    print(f"PGD ASR:        {asr_pgd*100:.2f}%")
    print(f"MI-FGSM ASR:    {asr_mifgsm*100:.2f}%")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # Using a smaller test batch size so memory doesn't overload during iterative attacks
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize Model and Optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train the model
    train(model, device, train_loader, optimizer, epochs=3)
    
    # Evaluate Clean Accuracy and Attacks
    test_attacks(model, device, test_loader, epsilon=0.3)