# Ollama

## (Optional) Set Up GCP Virtual Machine

### 1. Create A Static IP

We will need this for DNS later.

```
gcloud compute addresses create llama-server-ip --region=us-central1
```

### 2. Create VM
```
gcloud compute instances create llama-server \
    --machine-type=g2-standard-8 \
    --zone=us-central1-a \
    --boot-disk-size=200GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --accelerator=type=nvidia-l4,count=1 \
    --metadata="install-nvidia-driver=True" \
    --address=llama-server-ip
```

## Set Up Dns

We will serve our model over TLS. This requires that we have a domain.

### (Optional) Find GCP VM External IP

If you are using a GCP VM, then run this command to find the external IP address.

```
gcloud compute instances describe llama-server --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Set Up A Record

Update your DNS provider with an A record that points `ollama.example.com`, where `example.com` is your personal domain, to the external IP address.

We will be using Caddy's built-in support for Let's Encrypt to set up TLS for us using this domain name.

## Install cuda

### 1. Remove Previous CUDA Installations and Install the NVIDIA Driver

Ensure that there are no conflicting versions of CUDA or NVIDIA drivers installed, then install the NVIDIA driver version 535.

```bash
sudo apt-get --purge remove '*cublas*' 'cuda*' 'nsight*' 'nvidia*'
sudo apt-get autoremove
sudo apt-get install nvidia-driver-535
sudo reboot
```

### 2. Download and Install the CUDA Repository Pin

This step ensures that the CUDA repository takes precedence over other repositories.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

### 3. Download the CUDA Repository Package

Download the CUDA 12.2 repository package for Ubuntu 22.04.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
```

### 4. Install the CUDA Repository Package

Use `dpkg` to install the downloaded package.

```bash
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
```

### 5. Install the GPG Key

Install the GPG key to complete the setup.

```bash
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-216F19BD-keyring.gpg /usr/share/keyrings/
```

### 6. Update the Package Lists

Update the package lists to include the CUDA repository.

```bash
sudo apt-get update
```

### 7. Install CUDA Toolkit

Install the CUDA toolkit using `apt-get`.

```bash
sudo apt-get -y install cuda
```

### 8. Set Environment Variables

Ensure your environment variables are correctly set by adding the following lines to your `.bashrc` or `.zshrc`:

```bash
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 9. Verify the Installation

Run the following commands to verify that `nvidia-smi` and `nvcc` work correctly.

```bash
nvidia-smi
nvcc --version
```

[Source](https://forums.developer.nvidia.com/t/installing-cuda-on-ubuntu-22-04-rxt4080-laptop/292899)

## Install Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
```

[Source](https://ollama.com/download/linux)

Note: This should set up `ollama serve` in systemctl. Verify it is running using `sudo systemctl status ollama`

## Run Model

### 1. Download
Example: `ollama run deepseek-r1:32b --verbose`

Verify this model is working correctly by saying something in the prompt.

### 2. Test API

```
curl -v http://localhost:11434/api/generate -d '{
    "model": "deepseek-r1:32b",
    "prompt": "Hello!"
}'
```

If something is not working, check the logs using `journalctl -fxeu ollama`.

## Set Up Reverse Proxy

We want to protect our model so that only we can use it. Ollama does not have this functionality at the moment.

### 1. Install Caddy

```
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

### 2. Create Caddyfile

```
ollama.example.com {
    @authenticated {
        header X-API-Key "secret"
    }

    @unauthorized not header X-API-Key "secret"
    respond @unauthorized 401 {
        body "Unauthorized: Invalid or missing API key"
    }

    reverse_proxy localhost:11434 {
        header_up Host localhost
    }

    tls email@example.com
}
```

Important: Replace `secret` with a randomly generated value. Use `openssl rand -hex 32` or `uuidgen`.

Restart Caddy: `sudo systemctl restart caddy`

### 3. Test API

```
curl -v -H "X-API-Key: secret" https://ollama.hermanradtke.com/api/generate -d '{
    "model": "deepseek-r1:32b",
    "prompt": "Hello!"
}'
```

Note: Make sure to replace `secret` with the generated API key above.
