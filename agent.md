- Acesse todos os módulos e arquivos .py dentro de /neurapose_backend/ resuma ao máximo todos os comentários que explicam um pouco das linhas! Comentários de no máximo uma linha explicando o que uma linha ou o que uma fução faz! 

- Remova das primeiras linhas "# ========", deixe somente as linhas que tem o nome do arquivo resumido, por exemplo "# neurapose-backend/.../..."

- Dentro de "pre_processamento" tem uma função imprimir banner, gostaria de pegar ela como exemplo e usar o estilo dela dentro de "app". Organize melhor ela, deixe ela mais bonito esse banner o possivel! 

- Informações de terminal que aparecem no terminal do backend e nos terminais dentro do aplicativo, melhore os textos que são emitidos durantes os processamentos! 

- Banners de imformação em "app/testar_modelo" e "pre_processamento/processar":

==============================================================
PRÉ-PROCESSAMENTO — NEURAPOSE
==============================================================
YOLO              : [OK] yolov8l (se [ok]: cor verde; [erro]: cor vermelha)
TRACKER           : [OK] BoTSORT
OSNet ReID        : [OK] osnet_ain_x1_0_msmt17...
RTMPose-l         : [OK] rtmpose-l...
--------------------------------------------------------------
Processador       : <Buscar o processador automaticamente>
RAM               : <Buscar a RAM automaticamente>
GPU detectada     : <Buscar a GPU automaticamente>
VRAM              : <Buscar o VRAM do GPU automaticamente>
==============================================================

[NORMALIZAÇÃO] NORMALIZANDO VIDEO <FPS> FPS... (obs: se o valor configurado for diferente do fps do video, ele irá normalizar o video para o fps configurado, se for igual, ele irá pular essa etapa, "[NORMALIZAÇÃO] VIDEO NORMALIZADO EM <FPS> FPS.")

[YOLO] PROCESSANDO VIDEO...
[YOLO] 10 %
[YOLO] 20 %
[YOLO] 30 %
.... 
OU [YOLO] 10 % (UMA BARRA DE CARREGAMENTO SE POSSIVEL) 


==============================================================
TESTE DE MODELO — NEURAPOSE
==============================================================
YOLO              : [OK] yolov8l
TRACKER           : [OK] BoTSORT
OSNet ReID        : [OK] osnet_ain_x1_0_msmt17...
RTMPose-l         : [OK] rtmpose-l...
Modelo Temporal   : [OK] temporal fusion transformer "obs: (se usuário esclhou tft), Robust LStM (se usuário escolheu robust), BiLSTM (se usuário escolheu bilstm) e etc."
--------------------------------------------------------------
Processador       : <Buscar o processador automaticamente>
RAM               : <Buscar a RAM automaticamente>
GPU detectada     : <Buscar a GPU automaticamente>
VRAM              : <Buscar o VRAM do GPU automaticamente>
==============================================================