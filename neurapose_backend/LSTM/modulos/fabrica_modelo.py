# ================================================================
# LSTM/modulos/fabrica_modelo.py
# ================================================================

import torch
from pathlib import Path
import neurapose_backend.config_master as cm
from neurapose_backend.LSTM.models.models import (
    LSTM, RobustLSTM, PooledLSTM, BILSTM, AttentionLSTM,
    TCN, TransformerModel, TemporalFusionTransformer, WaveNet
)


class ClassifierFactory:
    def __init__(self, device):
        self.device = device
        
        # Hyperparâmetros base, onde input_size=C*V e T=TIME_STEPS (30)
        self.hp = dict(
            input_size=cm.NUM_CHANNELS * cm.NUM_JOINTS,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            dropout=0.3,
            num_classes=2,
            time_steps=cm.TIME_STEPS # 30
        )

    def _resolve_key(self, name: str) -> str:
        # Usa o nome do modelo (e.g., 'tft') para mapear para o nome da classe (e.g., 'temporalfusiontransformer')
        name = (name or "").strip().lower()
        aliases = {
            "lstm": "lstm", "simple-lstm": "lstm",
            "robust": "robustlstm", "robust-lstm": "robustlstm",
            "pooled": "pooledlstm", "pooled-lstm": "pooledlstm",
            "attention": "attentionlstm", "attention-lstm": "attentionlstm",
            "bilstm": "bilstm", "tcn": "tcn",
            "transformer": "transformer", "trans": "transformer",
            "tft": "temporalfusiontransformer",
            "wavenet": "wavenet", "wave": "wavenet",
        }
        # Tenta o alias, senão usa o nome da classe como fallback (ex: 'TemporalFusionTransformer' -> 'temporalfusiontransformer')
        key = aliases.get(name, name).replace('-', '') 
        return key

    def _safe_load(self, path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _infer_from_state_dict(self, state_dict):
        """
        Tenta inferir hidden_size, num_layers e num_classes a partir do state_dict.
        Atualiza self.hp com os valores encontrados.
        """
        # Lógica para LSTM / RNNs
        if "lstm.weight_hh_l0" in state_dict:
            # weight_hh_l0 tem shape [4 * hidden_size, hidden_size]
            w_hh = state_dict["lstm.weight_hh_l0"]
            self.hp["hidden_size"] = w_hh.shape[1]
            
            # Contar camadas
            layers = 0
            while f"lstm.weight_hh_l{layers}" in state_dict:
                layers += 1
            self.hp["num_layers"] = layers
            
        # Lógica para FC final (num_classes)
        if "fc.weight" in state_dict:
            self.hp["num_classes"] = state_dict["fc.weight"].shape[0]
        elif "classifier.weight" in state_dict:
             self.hp["num_classes"] = state_dict["classifier.weight"].shape[0]

    def _build_model(self, key: str):
        # Mapeamento do nome da chave para a classe (adaptado para TIME_STEPS, se necessário)
        
        # IMPORTANTE: Ajustar a hidden_size de saída para 30 nos modelos que usam 'flatten'
        # Seu modelo TFT/LSTM usa Pooling/Last State, então a lógica é simplificada.

        match key:
            case "tcn": return TCN(input_size=self.hp["input_size"], num_classes=2)
            case "wavenet": return WaveNet(input_size=self.hp["input_size"], num_classes=2)
            case "lstm": return LSTM(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"], self.hp["num_classes"])
            case "robustlstm": return RobustLSTM(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"], self.hp["dropout"])
            case "pooledlstm": return PooledLSTM(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"], self.hp["dropout"])
            case "attentionlstm": return AttentionLSTM(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"])
            case "bilstm": return BILSTM(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"], self.hp["dropout"])
            case "transformer": return TransformerModel(self.hp["input_size"], self.hp["hidden_size"], self.hp["num_layers"], self.hp["num_classes"], self.hp["num_heads"], self.hp["dropout"])
            case "temporalfusiontransformer": return TemporalFusionTransformer(
                input_size=self.hp["input_size"], d_model=self.hp["hidden_size"], n_heads=self.hp["num_heads"],
                num_encoder_layers=self.hp["num_layers"], num_decoder_layers=1,
                dropout=self.hp["dropout"], num_classes=self.hp["num_classes"]
            )
        raise ValueError(f"Modelo desconhecido: {key}")

    @staticmethod
    def load(model_dir: Path, device: str = "cpu"):
        """
        Método estático para carregar um modelo a partir de um diretório.
        Tenta extrair o tipo do modelo do nome do diretório ou usa config.
        """
        # Tenta extrair o tipo do modelo do nome do diretório
        # Padrão novo: <dataset>-modelo_<abbr>-acc_<float>
        # Padrão antigo: <model>-<dataset>
        
        name = model_dir.name
        if "-modelo_" in name:
            try:
                # Extrai o que está entre '-modelo_' e o próximo '-' (que deve ser o '-acc_')
                model_name = name.split("-modelo_")[1].split("-")[0]
            except:
                model_name = name.split("-")[0]
        else:
            # Fallback para o padrão antigo
            model_name = name.split("-")[0]
        
        factory = ClassifierFactory(device)
        model, mu, sigma = factory.load_model(model_name, model_dir)
        
        # Retorna apenas o que o testar_modelo espera (model, stats)
        # O testar_modelo espera: lstm_model, norm_stats 
        # e depois acessa norm_stats["mu"] e norm_stats["sigma"]
        # Vamos empacotar mu e sigma num dict para compatibilidade
        norm_stats = {}
        if mu is not None and sigma is not None:
             norm_stats = {"mu": mu, "sigma": sigma}
             
        return model, norm_stats

    def load_model(self, model_type: str, model_dir: Path):
        model_key = self._resolve_key(model_type)
        
        # Assume que o checkpoint é model_best.pt dentro do model_dir
        best_path = model_dir / "model_best.pt"
        norm_path = model_dir / "norm_stats.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado para o modelo {model_type.upper()} em: {best_path}")

        # 1. Carregar State Dict
        state_dict = self._safe_load(best_path)
        
        # 2. Inferir Hyperparâmetros
        self._infer_from_state_dict(state_dict)
        
        # 3. Construir e Carregar o Modelo
        model = self._build_model(model_key).to(self.device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Pode ocorrer se os HPs inferidos não baterem com o modelo salvo
            print(f"Aviso: Falha ao carregar state_dict (RuntimeError). Tentando strict=False. Erro original: {e}")
            model.load_state_dict(state_dict, strict=False)
            
        model.eval()

        # 4. Carregar Normalização
        mu, sigma = None, None
        if norm_path.exists():
            stats = self._safe_load(norm_path)
            mu, sigma = stats.get("mu"), stats.get("sigma")
            if mu is not None: mu = mu.to(self.device)
            if sigma is not None: sigma = sigma.to(self.device)

        return model, mu, sigma