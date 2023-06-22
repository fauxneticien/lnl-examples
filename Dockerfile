FROM fauxneticien/ptl2

RUN pip install lightning==2.0.2 \
    lhotse==1.14.0 \
    pandas \
    jiwer \
    hydra-core \
    wandb \
    torchinfo
