import torch
import torch.nn as nn
import pytest
import torch.nn.functional as F


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(LSTMFeatureExtractor, self).__init__()
        self.LSTM = nn.LSTM(input_size=2, hidden_size=16, num_layers=1,
                            batch_first=True, bidirectional=False)  # (seq_len, batch, input_size)

    def forward(self, s):
        output, (hid, cell) = self.LSTM(s)
        lstm_output = hid.view(hid.size(1), -1)  # => batch , layers * hid
        return lstm_output

def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out

class PolicyModule(nn.Module):
    def __init__(self, args):
        super(PolicyModule, self).__init__()
        self.device = 'cuda'

        self.output = 1
        self.feature_extractor = (16 * 1 * 1)

        self.last_hidden = self.feature_extractor * 2

        self.fc_layer1 = nn.Linear(self.feature_extractor, self.last_hidden)
        self.fc_layer2 = nn.Linear(self.last_hidden, self.last_hidden)
        self.fc_layer3 = nn.Linear(self.last_hidden, self.last_hidden)

        self.mu = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.sigma = NormedLinear(self.last_hidden, self.output, scale=0.1)
        self.normalDistribution = torch.distributions.Normal

    def forward(self, extract_states):
        fc_output1 = F.relu(self.fc_layer1(extract_states))
        fc_output2 = F.relu(self.fc_layer2(fc_output1))
        fc_output = F.relu(self.fc_layer3(fc_output2))
        # removed normalization of expected vaalue. Although the output is sending through the normalization, see inside normalization block. it is not done.
        # print("fc_output\n")
        # print(fc_output)
        mu = F.tanh(self.mu(fc_output))
        sigma = F.sigmoid(self.sigma(fc_output) + 1e-5)
        z = self.normalDistribution(0, 1).sample()
        action = mu + sigma * z
        action = torch.clamp(action, -1, 1)
        try:
            dst = self.normalDistribution(mu, sigma)
            log_prob = dst.log_prob(action[0])
        except ValueError:
            print('\nCurrent mu: {}, sigma: {}'.format(mu, sigma))
            print('shape: {}. {}'.format(mu.shape, sigma.shape))
            print(extract_states.shape)
            print(extract_states)
            log_prob = torch.ones(2, 1, device=self.device, dtype=torch.float32) * self.glucose_target
        return mu, sigma, action, log_prob
  
class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.FeatureExtractor = LSTMFeatureExtractor(args)
        self.PolicyModule = PolicyModule(args)

    def forward(self, s):
        lstmOut = self.FeatureExtractor.forward(s)
        mu, sigma, action, log_prob = self.PolicyModule.forward(lstmOut)
        return mu, sigma, action, log_prob
    
    def get_fim(self, x):
        mu, sigma, _, _ = self.forward(x)

        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(0)

        cov_inv = sigma.pow(-2).repeat(x.size(0), 1)

        param_count = 0
        std_index = 0
        id = 0
        std_id = id
        for name, param in self.named_parameters():
            if name == "sigma.weight":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1

        return cov_inv.detach(), mu, {'std_id': std_id, 'std_index': std_index}

@pytest.fixture
def setup():
    args = {
        'n_features': 2,
        'n_rnn_hidden': 16,
		'n_rnn_layers': 1,
		'bidirectional': False,
		'device': 'cuda',
		'n_action': 1,
		'rnn_directions': 1
    }

    batch_size = 32
    sequence_length = 10  # Number of time steps (you can adjust this as needed)
    input_size = 2  # Input size for each time step

    # Create sample inputs for the test
    states_batch = torch.randn((batch_size, sequence_length, input_size)).cuda()

    state_dim = 4002  # Set state_dim to 4002 based on the output from torch.randn_like(Jt)
    v = torch.randn((state_dim,), dtype=torch.float32).cuda()


    # Initialize the model
    actor_network = ActorNetwork(args).cuda()
    actor_network.eval()  # Switch to evaluation mode

    # Return the initialized objects for use in tests
    return actor_network, states_batch, v

def test_fvp_fim_runs(setup):
    actor_network, states_batch, v = setup
    
    def Fvp_fim(v):
        with torch.backends.cudnn.flags(enabled=False):
            M, mu, info = actor_network.get_fim(states_batch)
            #pdb.set_trace()
            mu = mu.view(-1)
            filter_input_ids = set([info['std_id']])

            t = torch.ones(mu.size(), requires_grad=True, device=mu.device)
            mu_t = (mu * t).sum()
            Jt = compute_flat_grad(mu_t, actor_network.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
            Jtv = (Jt * v).sum()
            Jv = torch.autograd.grad(Jtv, t)[0]
            MJv = M * Jv.detach()
            mu_MJv = (MJv * mu).sum()
            JTMJv = compute_flat_grad(mu_MJv, actor_network.parameters(), filter_input_ids=filter_input_ids, create_graph=True).detach()
            JTMJv /= states_batch.shape[0]
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
            return JTMJv + v * 0.1 # self.damping

    # Run the Fvp_fim function with the given v vector
    output = Fvp_fim(v)
    print(output)
    # Ensure that the output is not None
    assert output is not None

def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=True, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if (i in filter_input_ids):
            out_grads.append(torch.zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
        else:
            if (grads[j] == None):
                out_grads.append(torch.zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
            else:
                out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads