# Use an official Python runtime as a base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

# Set a non-interactive frontend (this prevents interactive prompts during package installation)
ENV DEBIAN_FRONTEND noninteractive

# Install any necessary dependencies (if required by PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    dssp \
    git \
    wget \
    unzip \
    vim \
    git-lfs \
    libglib2.0-0 \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Reset the frontend to its default mode
ENV DEBIAN_FRONTEND newt
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Assuming that the current directory (.) contains the PSLM repository contents
# and that it is being copied into the image at /app
COPY . /app
# otherwise:
#RUN git clone https://github.com/skalyaanamoorthy/PSLMs.git

# to get large files / datasets
RUN git lfs install --skip-smudge && \
    git lfs pull && \
    unzip ./data/preprocessed/msas.zip -d ./data/preprocessed/msas && \
    unzip ./data/preprocessed/weights.zip -d ./data/preprocessed/weights && \
    unzip ./data/rosetta_predictions.zip -d ./data/rosetta_predictions

# Create a virtual environment
RUN python3 -m venv /app/pslm

# Activate the virtual environment and install dependencies
RUN . /app/pslm/bin/activate && \
    pip install -r requirements.txt && \
    pip install evcouplings --no-deps && \
    pip install torch && \
    pip install -r requirements_inference.txt --no-deps

# COMMENT OUT tools you don't need or that break your installation

# AliStat for getting statistics from alignments (for analysis, not inference)
RUN git clone https://github.com/thomaskf/AliStat && \
    cd AliStat && \
    make

# ProteinMPNN SOA inverse-folding model (structural PSLM)
RUN git clone https://github.com/dauparas/ProteinMPNN

# Tranception MSA-dependent sequence PSLM
RUN git clone https://github.com/OATML-Markslab/Tranception && \
    curl -o Tranception_Large_checkpoint.zip https://marks.hms.harvard.edu/tranception/Tranception_Large_checkpoint.zip && \
    unzip Tranception_Large_checkpoint.zip && \
    rm Tranception_Large_checkpoint.zip

# statistical potential model (not PSLM)
RUN git clone https://github.com/chaconlab/korpm && \
    cd korpm/sbg && \
    sh ./compile_korpm.sh && \
    cd ../..

### NOTE: YOU MAY HAVE TO MODIFY THIS SECTION FOR YOUR PURPOSES ###
RUN wget https://salilab.org/modeller/10.5/modeller_10.5-1_amd64.deb && \
    env KEY_MODELLER=XXXX dpkg -i modeller_10.5-1_amd64.deb

RUN chmod +x convenience_scripts/append_modeller_paths.sh && \
    ./convenience_scripts/append_modeller_paths.sh
### NOTE: YOU MAY HAVE TO MODIFY THE SCRIPT ABOVE FOR YOUR SYSTEM ###

RUN chmod -R 777 /app

# Create the entrypoint script
RUN echo '#!/bin/bash\nsource /app/pslm/bin/activate\nexec "$@"' > /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the script as the default entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD /bin/bash
