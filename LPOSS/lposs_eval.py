def normalize_connection_graph(G):
    """
    图结构归一化处理 (对称归一化拉普拉斯矩阵构建)
    
    参数:
        G (csr_matrix): 原始邻接矩阵（稀疏格式）
        
    返回:
        Wn (csr_matrix): 归一化后的邻接矩阵，满足 Wn = D^{-1/2} W D^{-1/2}
    """
    W = csr_matrix(G)
    W = W - diags(W.diagonal(), 0)  # 移除自环
    S = W.sum(axis=1).A1  # 计算节点度向量 [N,]
    S[S == 0] = 1  # 防止除零错误
    D = cp.array(1.0 / cp.sqrt(S))  # D = diag(1/sqrt(degree))
    D[cp.isnan(D)] = 0  # 处理异常值
    D[cp.isinf(D)] = 0
    D_mh = diags(D.reshape(-1), 0)  # 构建对角阵
    Wn = D_mh @ W @ D_mh  # 对称归一化: Wn = D^{-1/2} W D^{-1/2}
    return Wn

def get_lposs_laplacian(feats, locations, height_width, sigma=0.0, pix_dist_pow=2, k=100, gamma=1.0, alpha=0.95, patch_size=16):
    """
    构建基于外观和空间信息的图拉普拉斯矩阵
    
    参数:
        feats (Tensor): DINO特征向量 [N, d]
        locations (Tensor): 各窗口左上角坐标 [B,4] (格式x1,y1,x2,y2)
        height_width (list): 各窗口特征图尺寸 [(h1,w1), (h2,w2), ...]
        sigma (float): 空间距离权重衰减系数
        pix_dist_pow (int): 空间距离计算的幂次（L2距离的平方）
        k (int): 每个节点的k近邻数
        gamma (float): 特征相似度锐化系数
        alpha (float): 标签传播中的平滑系数
        patch_size (int): DINO patch尺寸
        
    返回:
        L (csr_matrix): 图拉普拉斯矩阵 L = I - α * Wn
    """
    # === 空间位置编码 ===
    idx_window = torch.cat([w * torch.ones(h*w, device=feats.device, dtype=torch.int64) 
                          for w, (h, w_size) in enumerate(height_width)])  # 窗口索引向量
    idx_h = torch.cat([torch.arange(h).repeat_interleave(w) for h, w in height_width]).to(feats.device)  # 窗口内行坐标
    idx_w = torch.cat([torch.arange(w).repeat(h) for h, w in height_width]).to(feats.device)  # 窗口内列坐标
    
    # 计算绝对坐标 (中心点)
    loc_h = locations[idx_window, 1] + (patch_size // 2) + idx_h * patch_size  # [N,]
    loc_w = locations[idx_window, 0] + (patch_size // 2) + idx_w * patch_size  # [N,]
    locs = torch.stack((loc_h, loc_w), 1)  # [N,2]
    locs = locs.unsqueeze(0)  # 扩展批次维度
    
    # 计算空间距离矩阵
    dist = torch.cdist(locs, locs, p=2).squeeze(0)  # [N,N]
    dist = dist ** pix_dist_pow  # 距离幂次调整
    geometry_affinity = torch.exp(-sigma * dist)  # 高斯核转化
    
    # === 外观相似度计算 ===
    N = feats.shape[0]
    feats_np = feats.cpu().numpy().astype('float32')  # Faiss需要float32格式
    
    # 使用GPU加速的k近邻搜索
    res = faiss.StandardGpuResources()
    sims, knn_idx = faiss.knn_gpu(res, feats_np, feats_np, k, metric=faiss.METRIC_INNER_PRODUCT)
    sims = torch.from_numpy(sims).to(feats.device)
    knn_idx = torch.from_numpy(knn_idx).to(feats.device)
    
    # 相似度矩阵处理
    sims[sims < 0] = 0  # 删除负相似度
    sims = sims ** gamma  # 相似度锐化
    geo_affinity_sub = geometry_affinity.gather(1, knn_idx).flatten()  # 提取k近邻对应的空间权重
    
    # 构造稀疏邻接矩阵
    sims_flat = (sims.flatten() * geo_affinity_sub).cpu().numpy()
    rows = np.repeat(np.arange(N), k)
    cols = knn_idx.cpu().flatten().numpy()
    
    W = csr_matrix((sims_flat, (rows, cols)), shape=(N, N))
    W = W + W.T  # 对称化
    Wn = normalize_connection_graph(W)  # 归一化
    
    # 构建拉普拉斯矩阵
    L = eye(Wn.shape[0]) - alpha * Wn  # L = I - αWn
    return L

def dfs_search(L, Y, tol=1e-6, maxiter=10):
    """
    共轭梯度法求解线性系统 L x = Y
    
    参数:
        L (csr_matrix): 系数矩阵 (图拉普拉斯矩阵)
        Y (ndarray): 目标向量 (初始预测)
        
    返回:
        x (ndarray): 优化后的预测结果
    """
    # 转换到CuPy数组加速计算
    L_gpu = cp.sparse.csr_matrix(L)
    Y_gpu = cp.array(Y)
    x, _ = s_linalg.cg(L_gpu, Y_gpu, tol=tol, maxiter=maxiter)
    return x.get()

def perform_lp(L, preds):
    """
    多类别标签传播
    
    参数:
        L (csr_matrix): 图拉普拉斯矩阵 [N,N]
        preds (Tensor): 初始预测置信度 [N, C]
        
    返回:
        lp_preds (Tensor): 优化后的预测结果 [N, C]
    """
    lp_preds = cp.zeros(preds.shape)
    preds_np = cp.asarray(preds.cpu().numpy())
    
    # 逐类别优化
    for cls_idx in range(preds.shape[1]):
        Y = preds_np[:, cls_idx]
        lp_preds[:, cls_idx] = dfs_search(L, Y)
    
    return torch.as_tensor(lp_preds, device="cuda")

def get_pixel_connections(img, neigh=1):
    """
    构建像素级颜色相似性图 (用于LPOSS+)
    
    参数:
        img (Tensor): 输入图像 [1,3,H,W]
        neigh (int): 邻域范围（像素周围neigh×neigh区域）
        
    返回:
        rows (Tensor): 边起点索引
        cols (Tensor): 边终点索引
        pixel_pixel_data (Tensor): 边权重（颜色相似度）
        locs (Tensor): 像素坐标 [N,2]
    """
    # 颜色空间转换与归一化
    img_lab = rgb_to_lab(img[0]).permute(1,2,0)  # [H,W,3]
    img_lab = img_lab / torch.tensor([100, 128, 128], device=img.device)  # 归一化到[0,1]
    
    # 生成像素坐标网格
    H, W, _ = img_lab.shape
    img_flat = img_lab.reshape(H*W, 3)
    idx = torch.arange(H*W, device=img.device)
    loc_h = idx // W
    loc_w = idx % W
    locs = torch.stack([loc_h, loc_w], 1)  # [N,2]
    
    # 构建邻域连接关系
    rows, cols = [], []
    for dh, dw in product(range(-neigh, neigh+1), repeat=2):
        if dh == 0 and dw == 0: continue
        new_locs = locs + torch.tensor([dh, dw], device=img.device)
        
        # 边界掩码
        mask = (new_locs[:,0] >= 0) & (new_locs[:,0] < H) & (new_locs[:,1] >= 0) & (new_locs[:,1] < W)
        valid_rows = torch.where(mask)[0]
        valid_cols = new_locs[mask, 0] * W + new_locs[mask, 1]
        
        rows.append(valid_rows)
        cols.append(valid_cols)
    
    # 合并所有邻域连接
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    
    # 计算颜色相似度权重
    color_diff = (img_flat[rows] - img_flat[cols]).pow(2).sum(dim=1)
    return rows, cols, torch.sqrt(color_diff), locs

def get_lposs_plus_laplacian(img, preds, tau=0.1, neigh=6, alpha=0.95):
    """
    构建LPOSS+的拉普拉斯矩阵 (结合颜色相似性)
    
    参数:
        img (Tensor): 输入图像 [1,3,H,W]
        preds (Tensor): 初始预测置信度 [N,C]
        tau (float): 颜色相似性温度系数
        neigh (int): 邻域范围
        alpha (float): 平滑系数
        
    返回:
        L (csr_matrix): 图拉普拉斯矩阵
    """
    # 获取像素连接关系
    rows, cols, color_diff, _ = get_pixel_connections(img, neigh)
    
    # 计算颜色相似度权重
    weights = torch.exp(-color_diff / tau).cpu().numpy()
    
    # 构建稀疏邻接矩阵
    W = csr_matrix(
        (weights, (rows.cpu().numpy(), cols.cpu().numpy())),
        shape=(preds.size(0), preds.size(0))
    )
    Wn = normalize_connection_graph(W)
    L = eye(Wn.shape[0]) - alpha * Wn
    return L


class LPOSS_Infrencer(EncoderDecoder):
    def __init__(
            self,
            model,
            config,
            num_classes,
            test_cfg=dict(),
            **kwargs,
    ):
        """
        基于标签传播的开放词汇语义分割推理器
        
        参数:
            model (nn.Module): 特征提取主干网络（LPOSS模型）
            config (dict): 算法配置参数（sigma/gamma等超参数）
            num_classes (int): 实际目标类别数（包含背景）
            test_cfg (dict): 推理配置（模式/滑动窗口参数等）
        """
        super(EncoderDecoder, self).__init__()
        self.mode = test_cfg['mode']  # 推理模式 [whole|slide]
        self.num_classes = num_classes
        self.model = model  # LPOSS特征提取模型
        self.test_cfg = test_cfg  # 包含crop_size/stride等滑动窗口参数
        self.align_corners = False  # 插值对齐方式
        self.config = config  # 算法超参数配置

    @torch.no_grad()
    def encode_decode(self, img, meta_data):
        """双模态特征提取管道"""
        dino_feats, clip_feats, clf = self.model(img)
        return dino_feats, clip_feats, clf

    @torch.no_grad()
    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict], rescale) -> Tensor:
        """
        整图推理模式（适用于中等尺寸图像）
        
        流程:
            1. 提取全图DINO/CLIP特征
            2. 对齐特征图尺寸 & 归一化
            3. 计算CLIP初始预测
            4. 构建图拉普拉斯矩阵
            5. 标签传播优化预测
            6. (可选)像素级优化(LPOSS+)
            7. 上采样到原始分辨率
        """
        # 特征提取与尺寸对齐
        _, _, h_img, w_img = inputs.size()
        img_dino_feats, img_clip_feats, clf = self.encode_decode(inputs, batch_img_metas)
        
        # 调整CLIP特征图与DINO尺寸匹配
        if img_clip_feats.shape[1:3] != img_dino_feats.shape[1:3]:
            img_clip_feats = F.interpolate(img_clip_feats.permute(0,3,1,2), 
                                         size=img_dino_feats.shape[1:3], 
                                         mode='bilinear').permute(0,2,3,1)

        # 特征展平与归一化
        h, w, _ = img_dino_feats.shape[1:]  # 特征图空间维度
        dino_flat = img_dino_feats.reshape(h*w, -1)
        clip_flat = img_clip_feats.reshape(h*w, -1)
        dino_norm = F.normalize(dino_flat, dim=-1)
        clip_norm = F.normalize(clip_flat, dim=-1)

        # 初始CLIP预测
        clip_preds = clip_norm @ clf.T  # [N, num_classes]

        # 图构建与标签传播
        L = get_lposs_laplacian(
            dino_norm, 
            torch.zeros(1,4, device=inputs.device),  # 全图无位置偏移
            [(h, w)],  # 特征图尺寸
            sigma=self.config.sigma,  # 空间距离权重系数
            k=self.config.k,  # 节点连接数
            gamma=self.config.gamma,  # 相似度锐化因子
            patch_size=self.config.model.vit_patch_size
        )
        lp_preds = perform_lp(L, clip_preds)  # 优化后的预测

        # 结果重塑
        preds = lp_preds.view(h, w, -1).permute(2,0,1).unsqueeze(0)

        # LPOSS+像素级优化
        if self.config.pixel_refine:
            preds = self._pixel_refine(inputs, preds, h_img, w_img)

        # 类别数校正 & 上采样
        if preds.shape[1] > self.num_classes:
            preds = self.reduce_to_true_classes(preds)
        return resize(preds, batch_img_metas[0]['ori_shape'][:2], 
                     mode='bilinear', align_corners=self.align_corners)

    @torch.no_grad()
    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict], rescale) -> Tensor:
        """
        滑动窗口推理模式（适用于大尺寸图像）
        
        流程:
            1. 计算滑动窗口位置
            2. 批量提取窗口特征
            3. 拼接全局特征图
            4. 构建全局图结构
            5. 标签传播优化
            6. 滑窗结果融合
            7. (可选)像素级优化
            8. 上采样输出
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()

        # 计算滑动窗口网格
        h_grids = (h_img - h_crop + h_stride - 1) // h_stride + 1
        w_grids = (w_img - w_crop + w_stride - 1) // w_stride + 1
        locations = inputs.new_zeros((h_grids*w_grids, 4))  # 存储各窗口坐标[y1,y2,x1,x2]

        # 批量处理所有窗口
        images = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # 计算窗口坐标
                y1, x1 = h_idx*h_stride, w_idx*w_stride
                y2, x2 = min(y1+h_crop, h_img), min(x1+w_crop, w_img)
                y1, x1 = max(y2-h_crop, 0), max(x2-w_crop, 0)
                
                # 裁剪窗口并记录位置
                images.append(inputs[:, :, y1:y2, x1:x2])
                locations[h_idx*w_grids + w_idx] = torch.tensor([y1, y2, x1, x2])

        # 批量提取特征
        images = torch.cat(images)
        dino_feats, clip_feats, clf = self.encode_decode(images, None)
        
        # 调整CLIP特征尺寸
        if clip_feats.shape[1:3] != dino_feats.shape[1:3]:
            clip_feats = F.interpolate(clip_feats.permute(0,3,1,2), 
                                     dino_feats.shape[1:3], 
                                     mode='bilinear').permute(0,2,3,1)

        # 特征拼接与归一化
        dino_flat, hw_list = reshape_windows(dino_feats)  # 假设该函数将窗口特征拼接为全局布局
        clip_flat, _ = reshape_windows(clip_feats)
        dino_norm = F.normalize(dino_flat, dim=-1)
        clip_norm = F.normalize(clip_flat, dim=-1)
        clip_preds = clip_norm @ clf.T

        # 全局图构建与传播
        L = get_lposs_laplacian(
            dino_norm, 
            locations,  # 各窗口的绝对坐标
            hw_list,    # 各窗口的特征图尺寸
            sigma=self.config.sigma,
            k=self.config.k,
            gamma=self.config.gamma,
            patch_size=self.config.model.vit_patch_size
        )
        lp_preds = perform_lp(L, clip_preds)

        # 滑窗结果融合
        preds = inputs.new_zeros((1, clf.shape[0], h_img, w_img))
        count_mat = inputs.new_zeros((1, 1, h_img, w_img))
        idx_window = torch.cat([i*torch.ones(h*w, device=inputs.device) 
                              for i, (h,w) in enumerate(hw_list)])
        
        for win_id in range(len(hw_list)):
            # 获取当前窗口预测
            mask = idx_window == win_id
            win_pred = lp_preds[mask].view(*hw_list[win_id], -1).permute(2,0,1).unsqueeze(0)
            
            # 恢复窗口原始位置
            y1, y2, x1, x2 = locations[win_id].int().tolist()
            win_pred = resize(win_pred, (y2-y1, x2-x1), mode='bilinear')
            
            # 累加到全局预测图
            preds[..., y1:y2, x1:x2] += win_pred
            count_mat[..., y1:y2, x1:x2] += 1
        
        preds /= count_mat  # 重叠区域平均

        # LPOSS+优化 & 后处理
        if self.config.pixel_refine:
            preds = self._pixel_refine(inputs, preds, h_img, w_img)
        if preds.shape[1] > self.num_classes:
            preds = self.reduce_to_true_classes(preds)
        return resize(preds, batch_img_metas[0]['ori_shape'][:2],
                     mode='bilinear', align_corners=self.align_corners)

    def _pixel_refine(self, inputs, preds, h, w):
        """LPOSS+像素级优化子流程"""
        preds = preds[0].permute(1,2,0).view(h*w, -1)
        L = get_lposs_plus_laplacian(
            inputs, preds,
            tau=self.config.tau,       # 颜色相似度温度系数
            neigh=self.config.r//2,    # 邻域半径
            alpha=self.config.alpha    # 平滑系数
        )
        preds = perform_lp(L, preds).view(h, w, -1).permute(2,0,1).unsqueeze(0)
        return preds

    def reduce_to_true_classes(self, preds):
        """
        合并扩展的类别预测到实际类别数
        
        背景处理逻辑:
            - 前num_background_classes个通道为多种背景原型
            - 取这些通道的最大值作为最终背景类概率
            - 其他类别直接保留对应通道
        """
        num_bg = preds.shape[1] - self.num_classes + 1
        new_preds = torch.zeros_like(preds[:, :self.num_classes])
        new_preds[:, 1:] = preds[:, num_bg:]          # 保留目标类别
        new_preds[:, 0] = preds[:, :num_bg].max(1)[0] # 合并背景类
        return new_preds
