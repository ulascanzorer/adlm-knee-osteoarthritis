```mermaid

graph LR
    %% Styling for different components
    classDef encoder fill:#0065BD,stroke:#003359,stroke-width:1px,color:#fff
    classDef decoder fill:#0065BD,stroke:#003359,stroke-width:1px,color:#fff
    classDef latent fill:#E5E5E5,stroke:#0065BD,stroke-width:2px,color:#000
    classDef input fill:#FFFFFF,stroke:#333333,stroke-width:1px,color:#000
    classDef pain fill:#FFFFFF,stroke:#333333,stroke-width:1px,color:#000
    classDef truepain fill:#FFFFFF,stroke:#333333,stroke-width:1px,color:#000
    classDef loss fill:#005293,stroke:#003359,stroke-width:1px,color:#fff
    
    %% Input Layer
    Input["MRI Image<br/>160x224x224"]
    
    %% Encoder Layer
    Encoder[/"Encoder"/]

    %% Decoder Layer
    Decoder[\"Decoder"\]

    %% Output Layer
    Output["Reconstructed MRI Image<br/>160x224x224"]
    
    %% Latent Space (Bottleneck)
    Latent{{"Latent Space<br/> C = 64"}}

    %% Fully Connected Layer
    FC["KOOS Pain Head<br/>(AvgPool -> Linear)"]

    %% Predicted Pain Layer
    Pain["Predicted KOOS Pain"]

    %% Real Pain Layer
    TruePain["KOOS Pain Ground Truth"]

    %% Pain Loss
    PainLoss["KOOS Pain Loss"]

    %% Reconstruction Loss
    ReconLoss["Reconstruction Loss"]

    %% Final Loss
    FinalLoss["Final Loss"]
    
    %% Connections
    Input --> Encoder
    Input ==> ReconLoss
    Encoder --> Latent
    Latent --> FC
    Latent --> Decoder
    Decoder --> Output
    Output ==> ReconLoss
    FC --> Pain
    Pain ==> PainLoss
    TruePain ==> PainLoss
    PainLoss ==> FinalLoss
    ReconLoss ==> FinalLoss
    
    %% Apply styles
    class Input input
    class Encoder encoder
    class Latent latent
    class Decoder decoder
    class Output input
    class PainLoss loss
    class ReconLoss loss
    class FinalLoss loss
    class Pain pain
    class TruePain truepain
    class FC encoder
```