# REFORMER
- [REFORMER: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf)
- [Github of patrickvonplaten for Reformer](https://github.com/patrickvonplaten/notebooks)
- [Reformer in Huggingface](https://huggingface.co/transformers/model_doc/reformer.html)

#### 메모리를 적게 사용하기 위해 Reformer가 도입한 방법들
1. Axial Positional Encoding
2. LSH Attention (Locality-Sensitive Hashing Attention)
3. Reversible Residual Layer

## 1. Axial Positional Encoding
- 긴 input sequence가 주어지는 경우 standard positional encoding은 너무 많은 메모리를 사용함 
    - <img src="https://render.githubusercontent.com/render/math?math=d_h">: hidden size, ```config.hidden_size```
    - <img src="https://render.githubusercontent.com/render/math?math=n_{max}">: max position embeddings, ```config.max_position_embeddings``` *(defaults to 4096)*
    - Standard Positional Encoding: <img src="https://render.githubusercontent.com/render/math?math=n_{max} \times d_h">
    - Axial Positional Encoding: <img src="https://render.githubusercontent.com/render/math?math=n_\text{max}^1 \times d_h^1 + n_\text{max}^2 \times d_h^2">
        - <img src="https://render.githubusercontent.com/render/math?math=n_\text{max}^1 \times n_\text{max}^2 = n_\text{max}"> 
            
	    	- ```config.axial_pos_shape```: a tuple <img src="https://render.githubusercontent.com/render/math?math=(n_\text{max}^1, n_\text{max}^2)">
        - <img src="https://render.githubusercontent.com/render/math?math=d_h^1 + d_h^2 = d_h">
            
	    	- ```config.axial_pos_embds_dim```: a tuple <img src="https://render.githubusercontent.com/render/math?math=(d_h^1, d_h^2)">
            
### Example
<img src="https://render.githubusercontent.com/render/math?math=d_h = 4, n_{max} = 49">

#### 1. Standard Positional Encoding
![img](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/positional_encodings_default.png)
<br>
- 길이가 4인 49개의 벡터
- <img src="https://render.githubusercontent.com/render/math?math=4 \times 49">

#### 2. Axial Positional Encoding
- <img src="https://render.githubusercontent.com/render/math?math=n_\text{max}^1 =7, n_\text{max}^2 = 7">
- <img src="https://render.githubusercontent.com/render/math?math=d_h^1 = 1, d_h^2 = 3">
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/3d_positional_encoding.png"  width="400"/>
<br>
- 길이가 4인 49개의 벡터가 7개씩 7줄로 배열되어 있음 <img src="https://render.githubusercontent.com/render/math?math=(n_\text{max}^1 \times n_\text{max}^2)"> 


![alt text2](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/3d_positional_encoding_cut.png)
<br>
- 첫 번째 열의 7개 벡터만으로 나머지 벡터를 모두 나타내고자 함
- <img src="https://render.githubusercontent.com/render/math?math=e_{up} = d_{h}^2 = 3">
- <img src="https://render.githubusercontent.com/render/math?math=e_{down} = d_{h}^1 = 1">

<br><br>
- 7개 벡터의 윗부분(길이 3)은 column으로 확장함
    - 첫번째 열의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_1 ~ e_7">) 윗부분 3만큼은 모두 첫번째 열의 첫 번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_1">)의 윗부분으로 채워짐 
    - 두번째 열의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_8 ~ e_{14}">) 윗부분 3만큼은 모두 첫번째 열의 두 번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_2">)의 윗부분으로 채워짐
    - ...
    - 일곱번째 열의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_{43} ~ e_{49}">) 윗부분 3만큼은 모두 첫번째 열의 일곱번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_7">)의 윗부분으로 채워짐
<br><br>
- 7개 벡터의 아래부분(길이 1)은 row로 확장함
    - 첫번째 행의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_1, e_8, ..., e_{43}">) 아랫부분 1만큼은 모두 첫번째 열의 첫 번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_1">)의 아랫부분으로 채워짐
    - 두번째 행의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_2, e_9, ..., e_{44}">) 아랫부분 1만큼은 모두 첫번째 열의 두 번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_2">)의 아랫부분으로 채워짐  
    - ... 
    - 일곱번째 행의 벡터들(<img src="https://render.githubusercontent.com/render/math?math=e_7, e_{14}, ..., e_{49}">) 아랫부분 1만큼은 모두 첫번째 열의 일곱 번째 벡터(<img src="https://render.githubusercontent.com/render/math?math=e_7">)의 아랫부분으로 채워짐  
    
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/axial_pos_encoding.png"  width="500"/>

<br>

- <img src="https://render.githubusercontent.com/render/math?math=e'_1, e'2_, ..., e'_{49}"> 중에 같은 벡터가 하나도 없음
- <img src="https://render.githubusercontent.com/render/math?math=n_\text{max}^1 \times d_h^1 + n_\text{max}^2 \times d_h^2">만으로 positional encoding 가능


## 2. LSH Attention
- 기존 attention 계산에서 메모리 많이 사용하는 부분: <img src="https://render.githubusercontent.com/render/math?math=QK^T"> 
    - Q, K, T shape 모두 *\[batch size, length, <img src="https://render.githubusercontent.com/render/math?math=d_{model}">\]* 이라고 하면, <img src="https://render.githubusercontent.com/render/math?math=QK^T"> 의 shape은 *\[batch size, length, length\]*
        - ```config.attention_head_size```: <img src="https://render.githubusercontent.com/render/math?math=d_{model}"> 
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/1.png?raw=true" width="300"/>
<br>

#### 2.1. 하나의 query마다 attention 계산
-  attention을 matrix로 한 번에 계산하지 않고, 하나의 query 마다 계산한다면, <img src="https://render.githubusercontent.com/render/math?math=qK^T">의 shape은 *\[batch size, 1, length\]* 
    - only use memory proprtional to *length* 
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/2.png?raw=true" width="200"/>
<br>

#### 2.2. Q = K
- shared-QK Transformer
- 성능엔 큰 영향 없음
- first token 제외 자기 자신에게 attend하는 것 금지 
	-  Q = K 니까 항상 자기 자신과의 dot-product가 가장 값이 큼
    - cf. original Transformer는 자기 자신도 attend 가능 

#### 2.3. Hashing Attention
- <img src="https://render.githubusercontent.com/render/math?math=softmax(QK^T)">에서 어차피 가장 큰 값만 지배적인 역할을 하고 나머지는 큰 영향 주지 않음 
- <img src="https://render.githubusercontent.com/render/math?math=Q = K">이므로 각 <img src="https://render.githubusercontent.com/render/math?math=q_i">에 대해 가장 가까운 <img src="https://render.githubusercontent.com/render/math?math=key">들을 모두 찾아야 하는 것이 아니라, <img src="https://render.githubusercontent.com/render/math?math=q">끼리 가까운 것만 찾으면 됨 
- 유사한 <img src="https://render.githubusercontent.com/render/math?math=q">끼리 하나의 cluster로 모아 <img src="https://render.githubusercontent.com/render/math?math=m">개의 cluster를 만들 수 있음
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/4.png?raw=true" width="300"/>
<br>
- 그러면 같은 cluster 내에서만 softmax를 계산해도 됨. 유사하지 않은 모든 K(=Q)까지 모두 포함해서 계산할 필요 없음
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/5.png?raw=true" width="300"/>
<br>

#### LSH (Locality Sensitive Hashing)
- 유사한 벡터를 찾는 빠른 방법
- x → h(x)에 대응, 가까운 x끼리는 높은 확률로 같은 hash값을 갖게 됨 
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/7.png?raw=true"/>
<br>

- 위의 x,y는 random rotation 했을 때 다른 구역에 위치하는 경우가 많아서 높은 확률로 다른 hash 값을 갖게 됨 

- 아래 x,y는 random rotation 해도 계속 같은 구역에 위치해서 높은 확률로 같은 hash 값을 갖게 됨 

- hash 여러 번 해서 정확도 높임 
    - ```config.num_hashes```: 몇 번의 hash를 할 지 

<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/9.png?raw=true" width=600/>
<br>

- 같은 hash 값을 갖는 query끼리 하나의 bucket으로 배정하고 각 bucket끼리 모일 수 있도록 sorting
    - ```config.num_buckets```: 몇 개의 bucket으로 나눌지 
- sorting된 sequence를 chunk로 나누기 
    - chunk로 나누지 않으면 bucket마다 크기가 달라서 batch 처리 어려움 
    - ```config.lsh_attn_chunk_length```: chunk의 길이
- 자신이 속해 있는 chunk + 이전 chunk까지만 attend 
    - chunking 때문에 하나의 bucket이 다른 chunk로 쪼개질 수 있으니 이전 bucket까지 attend할 수 있어야 함
    - ```config.lsh_num_chunks_before```: 이전의 몇 개의 chunk까지 attend할 수 있게 할지 *(defaults to 1)*
    - ```config.lsh_num_chunks_after```: 이후의 몇 개의 chunk까지 attend할 수 있게 할지 *(defaults to 0)*
    
### 3. Reversible Residual Layer
- RevNet의 아이디어를 Transformer에 적용함 
- Reversible Layer + Chunking 으로 메모리 사용량이 layer 개수와 independent 해짐 


#### 3.1. RevNet to Transformer

#### RevNet
- model parameter만 사용하면서 특정 layer의 activation을 그 다음 layer의 activation으로 복원할 수 있음
- back propagation을 위해 모든 activation을 저장해놓을 필요가 없음
- reversible layer는 input, ouput을 pair로 받음  
    - input: <img src="https://render.githubusercontent.com/render/math?math=(x_1, x_2)">
    - output: <img src="https://render.githubusercontent.com/render/math?math=(y_1, y_2)">
    
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/13.png?raw=true" width=500/>
<br>

- substracting을 통해 output <img src="https://render.githubusercontent.com/render/math?math=(y_1, y_2)">만 갖고 input <img src="https://render.githubusercontent.com/render/math?math=(x_1, x_2)">을 복원할 수 있음 

<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/18.png?raw=true" width=500/>
<br>

#### Transformer
- RevNet의 F를 attention layer, G를 feed-forward layer로 대체 
- output of the last reversible transformer layer 만 저장하면 back propagation 가능 
<br>
<img src="https://github.com/yeounyi/reformer_summary/blob/main/img/14.png?raw=true" width=500/>
<br>

#### 3.2. Chunking Feed Forward Layer
- Attention 이후 2개의 Feed Forward Layer: <img src="https://render.githubusercontent.com/render/math?math=bf{Y}_{\text{out}} = \text{Linear}_{\text{out}}(\mathbf{Y}_\text{int}) = \text{Linear}_{\text{out}}(\text{Linear}_{\text{int}}(\mathbf{\overline{Z}}))">
    - 여기서 <img src="https://render.githubusercontent.com/render/math?math=i">번째 output인 <img src="https://render.githubusercontent.com/render/math?math=y_{out,i}">는 <img src="https://render.githubusercontent.com/render/math?math=i">번째 input에만 영향 받고 다른 위치에 있는 input에는 영향 받지 않음
    - 따라서 chunking 가능 
<br><br>  
- <img src="https://render.githubusercontent.com/render/math?math=Y_{int}">의 dimension이 <img src="https://render.githubusercontent.com/render/math?math=Y_{out}">의 dimension보다 큼. 더 많은 memory 필요
    - ```config.feed_forward_size```: <img src="https://render.githubusercontent.com/render/math?math=Y_{int}">의 output dimension
    - ```config.hidden_size```: <img src="https://render.githubusercontent.com/render/math?math=Y_{out}">의 output dimension
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/feed_forward_matrix.png" width=500/>
<br>

- chunking하여 계산 후 concat 하면 커다란 <img src="https://render.githubusercontent.com/render/math?math=Y_{int}"> matrix 전체를 다 저장해놓을 필요없어 memory 절약 가능
    - 그러나 시간은 좀 더 걸림 
    - memory와 time의 trade off 
<img src="https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/reformer_benchmark/chunked_feed_forward.png" width=500/>
<br> 
