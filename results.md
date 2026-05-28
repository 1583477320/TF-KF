# Human3.6M Model Evaluation Results

## MPJPE Comparison (Mean Per Joint Position Error)

The following table shows the MPJPE results for different models on the Human3.6M dataset.

```latex
\begin{table*}[htbp]
  \centering
  \caption{Average 3D joint error on Human3.6M for test subjects. The error is given in [mm].}
  \label{tab:h36m_results}
  \resizebox{\textwidth}{!}{
  \begin{tabular}{lcccccccccccccccc}
    \toprule
    Model & Directions & Discussion & Eating & Greeting & Phoning & Photo & Posing & Purchases & Sitting & SittingDown & Smoking & Waiting & WalkDog & Walking & WalkTogether & Mean \\
    \midrule
    kfl_QRFf_transformer & 111.15 & 117.38 & 127.44 & 133.03 & 141.74 & 138.87 & 139.07 & 139.83 & 140.93 & 131.45 & 126.31 & 115.62 & 124.12 & 121.13 & 114.05 & 199.22 \\
    Kalman & 161.70 & 185.12 & 186.61 & 195.82 & 204.43 & 211.37 & 164.56 & 180.88 & 253.07 & - & 184.30 & 198.28 & 213.58 & 203.75 & 207.67 & 200.11 \\
    kfl_QRFf & 112.74 & 117.90 & 130.67 & 136.13 & 144.89 & 141.56 & 141.74 & 141.95 & 142.75 & 133.22 & 127.20 & 116.65 & 125.13 & 122.19 & 114.87 & 199.88 \\
    \bottomrule
  \end{tabular}
  }
\end{table*}
```

### Key Notes:
- Error metric: MPJPE (Mean Per Joint Position Error)
- Unit: millimeters (mm)
- Lower values indicate better performance
