# ================================================================
# neurapose_backend/tracker/modulos/botsort.py
# ================================================================

import numpy as np
import torch
from colorama import Fore
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT as BOTSORT_ORIGINAL
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.gmc import GMC

import neurapose_backend.config_master as cm
from neurapose_backend.tracker.modulos.reid import CustomReID


class NumpyBoxesWrapper:
    """
    Wrapper que simula a interface do objeto Boxes do YOLO
    a partir de um numpy array no formato [x1, y1, x2, y2, conf, cls].
    Permite que a classe base BOTSORT funcione com detecções numpy.
    """
    def __init__(self, dets: np.ndarray):
        """
        Args:
            dets: numpy array (N, 6) com [x1, y1, x2, y2, conf, cls]
        """
        self._data = dets
        self._n = len(dets)
        
    def __len__(self):
        return self._n
    
    def __getitem__(self, idx):
        """Permite indexação: wrapper[0], wrapper[1:3], etc."""
        if isinstance(idx, (int, np.integer)):
            # Retorna uma linha como array 1D
            return self._data[idx]
        else:
            # Slice: retorna novo wrapper
            return NumpyBoxesWrapper(self._data[idx])
    
    def __iter__(self):
        """Permite iteração: for row in wrapper"""
        for i in range(self._n):
            yield self._data[i]
    
    @property
    def xyxy(self):
        """Retorna tensor com coordenadas [x1, y1, x2, y2]"""
        return torch.from_numpy(self._data[:, :4].astype(np.float32))
    
    @property
    def xywh(self):
        """Retorna tensor com coordenadas [cx, cy, w, h]"""
        x1, y1, x2, y2 = self._data[:, 0], self._data[:, 1], self._data[:, 2], self._data[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        return torch.from_numpy(xywh)
    
    @property
    def conf(self):
        """Retorna tensor com confiança"""
        return torch.from_numpy(self._data[:, 4].astype(np.float32))
    
    @property
    def cls(self):
        """Retorna tensor com classe"""
        return torch.from_numpy(self._data[:, 5].astype(np.float32))
    
    @property
    def data(self):
        """Retorna tensor com todos os dados"""
        return torch.from_numpy(self._data.astype(np.float32))


class CustomBoTSORT(BOTSORT_ORIGINAL):
    def __init__(self, frame_rate=30):
        class Args:
            pass

        args = Args()
        for k, v in cm.BOT_SORT_CONFIG.items():
            setattr(args, k, v)

        # Nao usamos o ReID nativo, usamos o nosso encoder
        args.with_reid = False

        super().__init__(args, frame_rate)
        self.args = args

        # GMC (motion compensation)
        self.gmc = GMC(method=args.gmc_method)

        # Custom ReID
        self.encoder = CustomReID(args.model)
        
        # Histórico para suavização EMA (Anti-Jitter)
        self.box_history = {}

    def get_kalmanfilter(self):
        return KalmanFilterXYWH()

    def update(self, results, img=None):
        """
        Sobrescreve update() para aceitar numpy.ndarray diretamente.
        Processa detecções numpy usando lógica interna do tracker.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Se já é um objeto com atributos nativos YOLO, passa para classe base
        if hasattr(results, 'conf') and hasattr(results, 'xyxy') and not isinstance(results, NumpyBoxesWrapper):
            return super().update(results, img)
        
        # Processa numpy array ou wrapper
        if isinstance(results, np.ndarray):
            dets = results
        elif isinstance(results, NumpyBoxesWrapper):
            dets = results._data
        else:
            dets = np.empty((0, 6))
            
        # DEBUG: Imprimir shape das detecções
        # print(f"DEBUG: Frame {self.frame_id} - dets shape: {dets.shape}")
        
        # Sem detecções: apenas Kalman predict
        if len(dets) == 0:
            for track in self.tracked_stracks:
                track.predict()
            for track in self.lost_stracks:
                track.mark_lost()
                if track.end_frame - track.start_frame > self.args.track_buffer:
                    removed_stracks.append(track)
            
            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]  # TrackState.Tracked
            self.lost_stracks = [t for t in self.lost_stracks if t not in removed_stracks]
            return self._format_output(self.tracked_stracks)
        
        # GMC (Global Motion Compensation)
        if self.args.gmc_method is not None and self.args.gmc_method != "none" and img is not None:
            try:
                # print("DEBUG: Applying GMC...")
                H = self.gmc.apply(img, dets[:, :4])
                for track in self.tracked_stracks:
                    track.apply_gmc(H)
                for track in self.lost_stracks:
                    track.apply_gmc(H)
            except Exception as e:
                print(f"[WARN] GMC falhou: {e}")
        
        # Inicializa novos tracks a partir das detecções
        detections = self.init_track(dets, img)
        # print(f"DEBUG: init_track returned {len(detections)} tracks")
        
        # Divide tracks em ativos e não confirmados
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        
        # Predição de todos os tracks
        for track in tracked_stracks:
            track.predict()
        
        # Matching: primeiro com tracks ativos
        dists = self.get_dists(tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        
        for itracked, idet in matches:
            track = tracked_stracks[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        
        # Segundo matching: tracks não matcheados vs detecções restantes (IOU)
        detections_second = [detections[i] for i in u_detection]
        r_tracked_stracks = [tracked_stracks[i] for i in u_track if tracked_stracks[i].state == 1]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track_2, u_detection_2 = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        
        for it in u_track_2:
            track = r_tracked_stracks[it]
            if not track.state == 0:  # Not Lost
                track.mark_lost()
                lost_stracks.append(track)
        
        # Terceiro matching: tracks não confirmados
        detections_left = [detections_second[i] for i in u_detection_2]
        dists = matching.iou_distance(unconfirmed, detections_left)
        matches, u_unconfirmed, u_detection_3 = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections_left[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # Inicializa novos tracks para detecções não matcheadas
        for inew in u_detection_3:
            det = detections_left[inew]
            if det.score < self.args.new_track_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)
        
        # Re-ativar tracks perdidos
        for track in self.lost_stracks:
            if track.end_frame < self.frame_id - self.args.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Atualiza listas
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # Remove duplicatas
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # --- POST-PROCESS: EMA SMOOTHING (COM PERSISTÊNCIA TTL) ---
        raw_tracks = self._format_output(self.tracked_stracks)
        
        # Configurações
        BOX_ALPHA = 0.15      # (15% Atual / 85% Histórico) - Suavização Pesada
        HISTORY_TTL = 60      # Mantém memória por 60 frames (2s) mesmo se o ID sumir

        smoothed_tracks = []
        current_ids_in_frame = set()

        # 1. Atualizar Tracks Atuais
        if len(raw_tracks) > 0:
            for i in range(len(raw_tracks)):
                track = raw_tracks[i].copy() # Trabalha em cópia
                
                # Tenta extrair ID e Coordenadas
                try:
                    t_id = int(track[4])
                    coords = track[:4]
                except:
                    smoothed_tracks.append(track)
                    continue

                current_ids_in_frame.add(t_id)

                if t_id in self.box_history:
                    # ID existe na memória: Recupera dados antigos
                    history_data = self.box_history[t_id]
                    
                    # Verifica compatibilidade (dict vs ndarray)
                    if isinstance(history_data, dict) and 'coords' in history_data:
                        prev_coords = history_data['coords']
                    else:
                        prev_coords = history_data # Fallback

                    # Aplica Fórmula EMA
                    smooth_coords = coords * BOX_ALPHA + prev_coords * (1.0 - BOX_ALPHA)
                    
                    # Atualiza memória e reseta contador de "sumiço"
                    self.box_history[t_id] = {'coords': smooth_coords, 'missed_frames': 0}
                    
                    # Aplica coordenadas suaves na saída
                    track[:4] = smooth_coords
                else:
                    # ID Novo: Inicializa sem suavizar (evita lag de entrada)
                    self.box_history[t_id] = {'coords': coords, 'missed_frames': 0}
                
                smoothed_tracks.append(track)
        else:
             smoothed_tracks = []

        # 2. Garbage Collection Inteligente (Remove apenas expirados)
        keys_to_remove = []
        # Itera sobre CÓPIA das chaves
        for t_id in list(self.box_history.keys()):
            if t_id not in current_ids_in_frame:
                # Recupera ou inicializa estrutura
                data = self.box_history[t_id]
                if not isinstance(data, dict): 
                    data = {'coords': data, 'missed_frames': 0}
                
                # Incrementa contador de frames perdidos
                data['missed_frames'] += 1
                self.box_history[t_id] = data
                
                # Se passou do limite (TTL), marca para remoção definitiva
                if data['missed_frames'] > HISTORY_TTL:
                    keys_to_remove.append(t_id)
        
        # Remove IDs mortos
        for k in keys_to_remove:
            del self.box_history[k]

        # Retorna Np Array (compatibilidade com yolo_stream)
        if len(smoothed_tracks) > 0:
            return np.array(smoothed_tracks)
        else:
            return np.empty((0, 7))
        # --- FIM SUAVIZAÇÃO EMA ---

    def init_track(self, results, img=None):
        if len(results) == 0:
            return []

        # Suporte a numpy.ndarray (vindo do yolo_stream.py) e Boxes (vindo do YOLO)
        if isinstance(results, np.ndarray):
            # Formato: [x1, y1, x2, y2, id, conf, cls] ou [x1, y1, x2, y2, conf, cls]
            if results.shape[1] >= 6:
                x1, y1, x2, y2 = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
                # Verifica se tem coluna de ID (7 colunas) ou não (6 colunas)
                if results.shape[1] >= 7:
                    confs = results[:, 5]
                    clss = results[:, 6]
                else:
                    confs = results[:, 4]
                    clss = results[:, 5]
                # Converte xyxy para xywh
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                bboxes = np.stack([cx, cy, w, h], axis=1)
            else:
                return []
        else:
            # Objeto Boxes do YOLO (tem atributos .xywh, .conf, .cls)
            bboxes = results.xywh.cpu().numpy()
            confs = results.conf.cpu().numpy()
            clss = results.cls.cpu().numpy()

        dets = np.hstack([bboxes, np.arange(len(bboxes)).reshape(-1, 1)])

        feats = self.encoder(img, dets)

        tracks = []
        for xywh, s, c, f in zip(bboxes, confs, clss, feats):
            # Ultralytics STrack exige vetor de tamanho 5 [x, y, w, h, score]
            element = np.append(xywh, s)
            tracks.append(BOTrack(element, s, c, f))

        return tracks

    def get_dists(self, tracks, detections):
        emb = matching.embedding_distance(tracks, detections)
        iou = matching.iou_distance(tracks, detections)

        alpha = self.args.appearance_thresh
        dists = emb * alpha + iou * (1.0 - alpha)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        return dists

    def _format_output(self, stracks):
        """
        Converte lista de STrack para numpy array no formato esperado:
        [x1, y1, x2, y2, track_id, conf, cls]
        """
        if not stracks:
            return np.empty((0, 7))
        
        outputs = []
        for t in stracks:
            if t.is_activated:
                # tlwh -> xyxy
                tlwh = t.tlwh
                x1 = tlwh[0]
                y1 = tlwh[1]
                x2 = tlwh[0] + tlwh[2]
                y2 = tlwh[1] + tlwh[3]
                outputs.append([x1, y1, x2, y2, t.track_id, t.score, t.cls])
        
        if not outputs:
            return np.empty((0, 7))
        
        return np.array(outputs)