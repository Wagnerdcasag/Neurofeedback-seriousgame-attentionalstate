n = input('1 = Gravação protocolo, 2 = Treino, 3 = Online 1 min, 4 = Retreino, 5 = Online de 5 min: ');
if n ==1
   E=input('Type of study select one option 1 = Evaluation of three BCIs (or modes), 2- Pilot study with a BCI (Mode 2): ');  
   if E==1 
      G=input('Select one option 1 = Group I, 2 = Group II, 3 = Group III: ');
      type_of_study=strcat('Evaluation of three BCIs (or modes)');
   elseif E==2 
      G=input('Select one option 1 = Group I, 2 = Group II: '); 
      type_of_study=strcat('Pilot study with a BCI (Mode 2)');
   end
   sham_mode = input('Select one option 0 = Intervention group, 1 = Sham group: ');
   Session_number = input(' Please provide the session number: ');
   Session_number=strcat('Session_',num2str(Session_number));
   Subject_ID = input(' Please provide the subject name or ID: ');
   if E==1
      Modo = input('Select one option 1 = BCI with active speed control, 2 = BCI with active speed control (using weighted mean), 3 = BCI without speed control: ');
   elseif E==2
      Modo=2;
   end
   
   
   if sham_mode==0
      intervation_or_sham_group=strcat('Intervention group');
   elseif sham_mode==1
      intervation_or_sham_group=strcat('Sham group');
   end
   
   if G==1 | G==2
      histerese=0;
   elseif G==3
      histerese=1;
   end
   dV_threshold=0.80;
   save('sham_mode.mat','sham_mode');
   save('dV_threshold.mat','dV_threshold');
   save('histerese.mat','histerese');
   save('intervation_or_sham_group.mat','intervation_or_sham_group');
   save('Session_number.mat','Session_number');
   save('Subject_ID.mat','Subject_ID');
   save('Modo.mat','Modo');
   save('type_of_study.mat','type_of_study');
   
     if Modo == 3
        minimal_time_sustained_attention = input(' Please define for players a minimal period of time (in seconds) to sustain his/her attention: ');
        VH=100; %Defining the high and low speeds
        VL=0;
        save('minimal_time_sustained_attention.mat','minimal_time_sustained_attention');
        save('VH.mat','VH');
        save('VL.mat','VL');
     end

end
  
switch n

%% Gravação Protocolo
    case 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    eletrodos = [1:19];
    V_TF = 0;
    Group = -1*ones(5,1);
    group_true = -1;
    x1 = 0;
    Fs = 500;
    janela = 2/(1/Fs);
    
    
    %% Comunicação com o unity
    client = Client(55001);

    %% instantiate the library
    disp('Loading the library...');
    lib = lsl_loadlib();

    % resolve a stream...
    disp('Resolving an EEG stream...');
    result = {};

    while isempty(result)
        result = lsl_resolve_byprop(lib,'type','EEG'); 
    end

    % create a new inlet
    disp('Opening an inlet...');
    inlet = lsl_inlet(result{1});

    disp('Now receiving data...');
    % Contador do vetor que vai comparar os 10 valores por segundo
    i = 1;

    % utilizado para salvar o matriz_ele_online e não sobrepor os valores mesmo
    % com o tamanho de chunk aleatorio
    aux = 0;

    %Contador para salvar energia
    contador = 1;

    %Parâmetros
    DATA_POINTS = janela;
    eletrodos_Total = 19;
    data_buffer = zeros(eletrodos_Total,DATA_POINTS);
    CHANNEL_OF_INTEREST = [1:19]; 
    SIZE_CHANNELS = size(CHANNEL_OF_INTEREST,2);
    
    % variaveis para mandar velocidade
    
    b = 0;
    nSteps = 5;
    ref = cputime;
    
    
    while 1
        [vec,ts] = inlet.pull_chunk();    
        new_points = vec(1:eletrodos_Total,:);    
        new_length = size(new_points,2);

        % Guardar os valores dentro da matriz para posterior processamento
        matriz_ele_online(:,(aux+1):(new_length+aux)) = new_points;
        aux = new_length + aux;

        % Atualização do Buffer
        %data_buffer(1:eletrodos_Total,1:DATA_POINTS-new_length) = data_buffer(1:eletrodos_Total,new_length+1:end);
        %data_buffer(1:eletrodos_Total,DATA_POINTS-new_length+1:end) = new_points(1:eletrodos_Total,:);
        
   
        
        %X_E = data_buffer(eletrodos,:)';

        
    %% Mandando a velocidade pro unity
     
     elapsed = cputime - ref;
     
     if elapsed > 1.0
         % Sends an alternating sequence of 0s and 1s
            if b > nSteps
                b = 0;
            end
            
            client.send(100.0 / nSteps * b + 0.2177716123);
            b = b+1;
            
            ref = cputime;
     end
     
    end
    
    %FAZER MANUALMENTE
    matriz_eletrodos_total = matriz_ele_online';
    save('matriz_eletrodos_total.mat','matriz_eletrodos_total');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[matriz_ele_online] = gravacao_protocolo_chunk()';
    %matriz_eletrodos_total = matriz_ele_online;
    %[matriz_eletrodos_total] = gravacao_protocolo_chunk(1,300,20)';
    cd1=cd;
    if  isdir('Database_BCI_Attention')==0
        mkdir('Database_BCI_Attention');
    end
    cd('Database_BCI_Attention');
    
    if  isdir(type_of_study)==0
        mkdir(type_of_study);
    end
    cd(type_of_study);
    
    if  isdir(intervation_or_sham_group)==0
        mkdir(intervation_or_sham_group);
    end
    cd(intervation_or_sham_group);
    
    if  isdir(Session_number)==0
        mkdir(Session_number);
    end
    cd(Session_number);
    
    if  isdir(Subject_ID)==0
        mkdir(Subject_ID);
    end
    cd(Subject_ID);
    if Modo ==3
       filename=strcat('minimal_time_sustaining_attention_',Subject_ID,'_M',num2str(Modo),'.mat');
       save(filename,'minimal_time_sustained_attention'); 
    end
    if  isdir('Phase_1_non_active_training')==0
        mkdir('Phase_1_non_active_training');
    end
    cd('Phase_1_non_active_training');
    filename=strcat('matriz_eletrodos_total_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'matriz_eletrodos_total');
    cd(cd1);
    
   
    
%% Primeiro treino
    case 2
        
    clear
    clc

    cd1=cd;
    load('intervation_or_sham_group.mat');
    load('Session_number.mat');
    load('Subject_ID.mat');
    load('Modo.mat');
    load('type_of_study.mat')
    if Modo ==3
         load minimal_time_sustained_attention
    end
    cd('Database_BCI_Attention');
    cd(type_of_study);
    cd(intervation_or_sham_group);
    cd(Session_number);
    cd(Subject_ID);
    cd('Phase_1_non_active_training');
    filename=strcat('matriz_eletrodos_total_',Subject_ID,'_M',num2str(Modo),'.mat');
    load (filename);
    cd(cd1)
 
    %% Treino
    % Variaveis importantes
    Fs = 500;
    band = [1 30];
    num_trial = 1;
    janela = 2/(1/Fs);
    overlap = 0.1/(1/Fs);
    CH = [1:19];
    bloco = 30;
    classe_1 = 0;
    classe_2 = 1;

    % Montando as matrizes para o treino(atencao e relaxado)
    for trial = 1:1:5
        pulo = (trial - 1) * bloco/(1/Fs); 

        for tempo = [1:overlap:15/(1/Fs) - janela] + pulo

            T = [tempo:janela+(tempo-1)];
            S_impar = T + classe_1 * 15/(1/Fs);
            S_par   = T + classe_2 * 15/(1/Fs);

            dados_atencao_treino(:,:,num_trial) = fftfilterv2(matriz_eletrodos_total(S_impar,CH),Fs,band)';
            dados_relaxado_treino(:,:,num_trial) = fftfilterv2(matriz_eletrodos_total(S_par,CH)  ,Fs,band)';

            num_trial = num_trial + 1;
        end
    end
    
    %Permutação
     X_T_final = [permute(dados_atencao_treino,[1 3 2])  permute(dados_relaxado_treino,[1 3 2])];
     X_T_final = permute(X_T_final,[1 3 2]);

    [ppppp,pppppp,p]  = size(dados_atencao_treino);
    [ppp,pppp,p1] = size(dados_relaxado_treino);

    class = ones(p,1); 
    class(p+1:p1+p,1) = -1;

    % Encontrando as características utilizando o método Riemannian 
    Ctrain = covariances(X_T_final);
    Ctemp = mean_covariances(Ctrain,'riemann');  
    full_training_set = Tangent_space(Ctrain,Ctemp)';

    %Média e Desvio Padrão do set de treino
    Media_treino = mean(full_training_set);
    Desvio_padrao_treino = std(full_training_set);
    full_training_set = (full_training_set - Media_treino)./ Desvio_padrao_treino;

    %% Treinamento classificador
    SVMModel = fitcsvm(full_training_set,class,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

    %% Criando o Modelo do Classificador
    CVSVMModel = crossval(SVMModel);

    %% Estimar a taxa de acurácia 
    classLoss = kfoldLoss(CVSVMModel);
    Acuracia_treino = 100 - classLoss*100

    %% Plot dados
    plot_PCA(full_training_set,class)     

    %% Teste
    num_trial = 1;
    spatial_filter=1; %Appying CAR 
    theta_band=[4 7];
    beta_band=[13 30];
    PB_method='fft';
    clear dados_atencao
    clear dados_relaxado
    
    
    clear PSD_dados_atencao_theta;
     clear PSD_dados_atencao_beta;
     clear PSD_dados_relaxado_theta;
     clear PSD_dados_relaxado_beta;

    for trial = 6:1:10
        pulo = (trial - 1)*bloco/(1/Fs); 

        for tempo = [1:overlap:15/(1/Fs) - janela] + pulo

            T = [tempo:janela+(tempo-1)];
            S_impar = T + classe_1*15/(1/Fs);
            S_par = T + classe_2*15/(1/Fs);

            dados_atencao_teste (:,:,num_trial) = fftfilterv2(matriz_eletrodos_total(S_impar,CH),Fs,band)';
            dados_relaxado_teste(:,:,num_trial) = fftfilterv2(matriz_eletrodos_total(S_par,CH)  ,Fs,band)';
            
            
             media_impar = 0;
             media_par = 0;
             if spatial_filter==1
                media_impar = mean(matriz_eletrodos_total(S_impar,CH),2);
                media_par   = mean(matriz_eletrodos_total(S_par,CH),2);
             end
            
            dados_atencao_CAR  = fftfilterv2(matriz_eletrodos_total(S_impar,CH) - media_impar,Fs,band);
            dados_relaxado_CAR = fftfilterv2(matriz_eletrodos_total(S_par,CH) - media_par,Fs,band);
            
            
            PSD_dados_atencao_theta (num_trial,:) = powerband_enrique(dados_atencao_CAR,hamming(500),500/2,Fs,theta_band,PB_method);
            PSD_dados_relaxado_theta (num_trial,:) = powerband_enrique(dados_relaxado_CAR,hamming(500),500/2,Fs,theta_band,PB_method);

            PSD_dados_atencao_beta (num_trial,:) = powerband_enrique(dados_atencao_CAR,hamming(500),500/2,Fs,beta_band,PB_method);
            PSD_dados_relaxado_beta (num_trial,:) = powerband_enrique(dados_relaxado_CAR,hamming(500),500/2,Fs,beta_band,PB_method);

            

             num_trial = num_trial + 1;
        end
    end

    clear X_T_final

    %Permutação
     X_T_final = [permute(dados_atencao_teste,[1 3 2])  permute(dados_relaxado_teste,[1 3 2])];
     X_T_final = permute(X_T_final,[1 3 2]);

    [pp,ppp,p]  = size(dados_atencao_teste);
    [pp1,ppp1,p1] = size(dados_relaxado_teste);

    clear class
    class = ones(p,1); 
    class(p+1:p1+p,1) = -1;

    % Encontrando as características utilizando o método Riemannian 
    Ctrain = covariances(X_T_final);
    full_testing_set = Tangent_space(Ctrain,Ctemp)';
    full_testing_set =(full_testing_set - Media_treino)./Desvio_padrao_treino;

    [Group,groupgroup] = predict(SVMModel,full_testing_set);
    Acuracia_teste = sum(Group == class) / length(class) * 100
    plot_PCA(full_testing_set,class) 

    %% velocidade
    
    full_atencao = find(Group(1:p)==1); %p is the total attention patterns 
    full_desatencao = p+find(Group(p+1:end)==-1);
    
    
    clear theta_beta_ratio_attetion_T1;
    clear theta_beta_ratio_non_attetion_T1;
     
    theta_beta_ratio_attetion_T1=PSD_dados_atencao_theta./PSD_dados_atencao_beta;
    theta_beta_ratio_non_attetion_T1=PSD_dados_relaxado_theta./PSD_dados_relaxado_beta;
     
    theta_beta_ratio=[theta_beta_ratio_attetion_T1;theta_beta_ratio_non_attetion_T1];
     
%      EEG_labels_plot = {'F7','Fp1','Fp2','F8','F3','Fz',...
%                      'F4','C3','Cz','P8','P7','Pz',...
%                      'P4','T3','P3','O1','O2','C4',...
%                      'T4'};

  

   frontal_lobe=[1:7]; %F7,Fp1,Fp2,F8,F3,Fz,F4
   temporal_lobe=[14,19]; %T7,T8
   
   TB_A_T1_F=mean(theta_beta_ratio(full_atencao,frontal_lobe),2);
  
   TB_NA_T1_F=mean(theta_beta_ratio(full_desatencao,frontal_lobe),2);   
   
   TB_A_T1_T=mean(theta_beta_ratio(full_atencao,temporal_lobe),2);
  
   TB_NA_T1_T=mean(theta_beta_ratio(full_desatencao,temporal_lobe),2); 
   
   TB_A_T1_FT=mean(theta_beta_ratio(full_atencao,[frontal_lobe temporal_lobe]),2);
  
   TB_NA_T1_FT=mean(theta_beta_ratio(full_desatencao,[frontal_lobe temporal_lobe]),2);  
   
   TB_A_T1=[TB_A_T1_F TB_A_T1_T TB_A_T1_FT];
   TB_NA_T1=[TB_NA_T1_F TB_NA_T1_T TB_NA_T1_FT];
   
   
   [III,I]=max([abs(mean(TB_A_T1_F)-mean(TB_NA_T1_F)) abs(mean(TB_A_T1_T)-mean(TB_NA_T1_T)) abs(mean(TB_A_T1_FT)-mean(TB_NA_T1_FT))]);
   
    
    weights_attention=1*(2-Modo)/length(full_atencao)+(Modo-1)*(1./TB_A_T1(:,I))./sum((1./TB_A_T1(:,I)));
    media_atencao = sum(full_testing_set(full_atencao,:).*weights_attention);
    weights_non_attention=1*(2-Modo)/length(full_desatencao)+(Modo-1)*TB_NA_T1(:,I)./sum(TB_NA_T1(:,I));
    media_desatencao = sum(full_testing_set(full_desatencao,:).*weights_non_attention);

    % Distancia Euclidiana entre os centroids
    dist_euc = sum((media_desatencao - media_atencao).^2).^0.5;

    % Pegando os dados que foram classificados corretamente como atenção e
    % desatenção
    recognized_attention_set = full_testing_set(full_atencao,:);
    recognized_non_attention_set = full_testing_set(full_desatencao,:);

    % Calculando as distancias entre o centroid e todos os pontos da atenção
    clear dist_centroid_atencao;
    for i = 1:1:length(full_atencao)
       dist_centroid_atencao(i) = sum((media_atencao - recognized_attention_set(i,:) ).^2).^0.5;

    end

    % Calculando as distancias entre o centroid e todos os pontos da desatenção
    clear dist_centroid_desatencao
    for i = 1:1:length(full_desatencao)
       dist_centroid_desatencao(i) = sum((media_desatencao - recognized_non_attention_set(i,:) ).^2).^0.5;

    end

    %Limitando os dados entre Lim_Inf e Lim_Sup
    Quant_val = quantile(dist_centroid_atencao,[0.25 0.50 0.75 1]);
    Lim_Sup = Quant_val(3) + 1.5*(Quant_val(3) - Quant_val(1));
    Lim_Inf = Quant_val(3) - 1.5*(Quant_val(3) - Quant_val(1));
    I = find(dist_centroid_atencao > Lim_Inf & dist_centroid_atencao < Lim_Sup);
    dist_centroid_atencao = dist_centroid_atencao(I);

    Quant_val_des = quantile(dist_centroid_desatencao,[0.25 0.50 0.75 1]);
    Lim_Sup_des = Quant_val_des(3) + 1.5*(Quant_val_des(3) - Quant_val_des(1));
    Lim_Inf_des = Quant_val_des(3) - 1.5*(Quant_val_des(3) - Quant_val_des(1));
    I_des = find(dist_centroid_desatencao > Lim_Inf_des & dist_centroid_desatencao < Lim_Sup_des);
    dist_centroid_desatencao = dist_centroid_desatencao(I_des);

    %% Valores das Velocidades
    %For attention level
    V_min = 40;
    V_med = 70;
    V_max = 100;
    %For non_attention condition
    V_min_des = 0;
    V_med_des = 20;
    V_max_des = 40;

    V = interp1([Lim_Sup Quant_val(2) Lim_Inf],[V_min V_med V_max],dist_centroid_atencao,'cubic');
    P = polyfit(dist_centroid_atencao,V,3);
    V_TF = polyval(P,dist_centroid_atencao);

    V_des = interp1([Lim_Inf_des Quant_val_des(2) Lim_Sup_des],[V_min_des V_med_des V_max_des],dist_centroid_desatencao,'cubic');
    P_des = polyfit(dist_centroid_desatencao,V_des,3);
    V_TF_des = polyval(P_des,dist_centroid_desatencao);

    figure;
    plot(dist_centroid_atencao,V_TF,'ro')
    hold on;
    plot(dist_centroid_desatencao,V_TF_des,'bo')
    
    %Saving data
    
    save('SVMModel.mat','SVMModel')
    save('janela.mat','janela')
    save('Ctemp.mat','Ctemp')
    save('dados_atencao_treino.mat','dados_atencao_treino')
    save('dados_atencao_teste.mat','dados_atencao_teste')
    save('dados_relaxado_treino.mat','dados_relaxado_treino')
    save('dados_relaxado_teste.mat','dados_relaxado_teste')
    save('V_min.mat','V_min')
    save('V_max.mat','V_max')
    save('V_min_des.mat','V_min_des')
    save('V_max_des.mat','V_max_des')
    save('Lim_Sup.mat','Lim_Sup');
    save('Lim_Inf_des.mat','Lim_Inf_des');
    save('media_atencao.mat','media_atencao')
    save('media_desatencao.mat','media_desatencao')
    save('P.mat','P')
    save('P_des.mat','P_des')
    save('Media_treino.mat','Media_treino')
    save('Desvio_padrao_treino.mat','Desvio_padrao_treino')
    save('theta_beta_ratio.mat','theta_beta_ratio');
    
    

%%   Online 1 min
    case 3
        
    clear 
    close all
    cd1=cd;
    load('intervation_or_sham_group.mat');
    load('Session_number.mat');
    load('Subject_ID.mat');
    load('Modo.mat');
    load('sham_mode.mat');
    load('type_of_study.mat')
    
    load SVMModel
    load Ctemp
    load janela
    load V_min
    load V_max
    load V_min_des
    load V_max_des
    load Lim_Sup
    load Lim_Inf_des
    load P
    load P_des
    load media_atencao
    load media_desatencao
    load Media_treino
    load Desvio_padrao_treino
    
    VH=V_max;
    VL=V_min;
    
    if Modo==3
        load VH
        load VL
        load minimal_time_sustained_attention  
        BCI_latency = 0;
    end
    
 
    eletrodos = [1:19];
    V_TF = 0;
    
    velocidade = 0;
    band = [1 30];
    vconst = 0;
    Group = -1*ones(5,1);
    group_true = -1;
    Fs = 500;
    k = 19;
    x = 0;
    x1 = 0;
   
    %% Comunicação com o unity
    client = Client(55001);

    %% instantiate the library
    disp('Loading the library...');
    lib = lsl_loadlib();

    % resolve a stream...
    disp('Resolving an EEG stream...');
    result = {};

    while isempty(result)
        result = lsl_resolve_byprop(lib,'type','EEG'); 
    end

    % create a new inlet
    disp('Opening an inlet...');
    inlet = lsl_inlet(result{1});

    disp('Now receiving data...');
    % Contador do vetor que vai comparar os 10 valores por segundo
    i = 1;

    % utilizado para salvar o matriz_ele_online e não sobrepor os valores mesmo
    % com o tamanho de chunk aleatorio
    aux = 0;

    %Contador para salvar energia
    contador = 1;

    %Parâmetros
    DATA_POINTS = janela;
    eletrodos_Total = 19;
    data_buffer = zeros(eletrodos_Total,DATA_POINTS);
    CHANNEL_OF_INTEREST = [1:19]; 
    SIZE_CHANNELS = size(CHANNEL_OF_INTEREST,2);
    while 1
        [vec,ts] = inlet.pull_chunk();    
        new_points = vec(1:eletrodos_Total,:);    
        new_length = size(new_points,2);

        % Guardar os valores dentro da matriz para posterior processamento
        matriz_ele_online(:,(aux+1):(new_length+aux)) = new_points;
        aux = new_length + aux;

        % Atualização do Buffer
        data_buffer(1:eletrodos_Total,1:DATA_POINTS-new_length) = data_buffer(1:eletrodos_Total,new_length+1:end);
        data_buffer(1:eletrodos_Total,DATA_POINTS-new_length+1:end) = new_points(1:eletrodos_Total,:);

    %% aux > quantidade de amostras por segundo , tbm mudar o DATA_POINTS para a quantidade de amostras    
        if(aux>janela) 
            
          if x1==0 %& Modo==3%First epoch
             tic; %e isso por qué aqui????????
          end
   
        
        X_E = fftfilterv2(data_buffer(eletrodos,:)',Fs,band)';

        Ctest = covariances(X_E); 
        validation_set = Tangent_space(Ctest,Ctemp)'; 
        validation_set = (validation_set - Media_treino)./Desvio_padrao_treino;

    % Classificação
        Group(1:end-1) = Group(2:end);
        [Groupp,groupgroup1] = predict(SVMModel,validation_set);
        Group(end) = Groupp;
       

    %% Mandando a velocidade pro unity 
    if sum(Group==mode(Group))== length(Group) && mode(Group)==1
       group_true = 1;

    elseif sum(Group==mode(Group))== length(Group) && mode(Group)==-1
       group_true = -1;

    end
    
    
           %sham_mode=1 to facilite the flight speed control with very low
           %attention, 0 otherwise
          if group_true == 1 && Group(end)==1
              x = x + 1;
              X_E_online_1(:,:,x) = X_E;

             if Modo==3 
                 %The flight speed is controlled taking into account the sustained attention 
                velocidade = velocidade + group_true*(BCI_latency/minimal_time_sustained_attention)*100 + group_true*sham_mode*10;
             else
               %The flight speed is actively controlled
               dist_euc = sum((media_atencao - validation_set).^2).^0.5;
               velocidade = polyval(P,dist_euc)+sham_mode*40; %sham_mode=1 to facilite the flight speed control with very 
               VH=V_max;                                      %low attention
               VL=V_min;
             end
      
       
        elseif group_true == -1 && Group(end) == -1
           if Modo==3 
              %The flight speed is controlled taking into account the sustained attention   
              velocidade = velocidade + group_true*(BCI_latency/minimal_time_sustained_attention)*100 + group_true*sham_mode*10;
           else
              %The flight speed is actively controlled
              dist_euc = sum(( media_desatencao - validation_set).^2).^0.5;
              velocidade = polyval(P_des,dist_euc)+ sham_mode*40;   %sham_mode=1 to facilite the flight speed control with very
              VH=V_max_des;                                         %low attention
              VL=V_min_des; 
           end

        end
    
       %Comparing with histerese for maintaining maximum speed for a defined condition 
        if velocidade < VL 
           V_TF = VL;

        elseif velocidade > VH | (sham_mode == 1 & velocidade > VL & V_TF == VH)
           V_TF = VH;
        else
           V_TF = velocidade 
        end
        
    client.send(V_TF)

       if x1==0 %& Modo==3  %First epoch
           BCI_latency = toc;
       end
    
    %Saving all data
    x1 = x1 + 1;
    X_E_online_1_all(:,:,x1) = X_E;
    all_labels_1(x1,1)=group_true;
    all_attention_decisions_1(x1,1)=(1+group_true)*(group_true*Group(end)+1)/4; %given 1 for attention decisions, and 0 otherwise
    all_V_TF_1(x1,1)=V_TF;
    
    

       end

    end
    
%*************************************************************************
%    REMOVE COMMENTS TO SAVE DATA
%*************************************************************************

   
%     cd1=cd;
%     load('dV_threshold.mat');
%     V_TF_up_to_40_mean=mean(all_V_TF_1(find(all_V_TF_1>40)));
%     V_TF_award=V_TF_up_to_40_mean+(100-V_TF_up_to_40_mean)*dV_threshold;
%     save('V_TF_award.mat','V_TF_award');
%     cd('Database_BCI_Attention');
%     cd(type_of_study);
%     cd(intervation_or_sham_group);
%     cd(Session_number);
%     cd(Subject_ID);
%     if  isdir('Phase_2_active_training')==0
%         mkdir('Phase_2_active_training');
%     end
%     cd('Phase_2_active_training');
%     filename=strcat('X_E_online_1_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'X_E_online_1');
%     filename=strcat('X_E_online_1_all_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'X_E_online_1_all');
%     filename=strcat('all_labels_1_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'all_labels_1');
%     filename=strcat('all_attention_decisions_1_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'all_attention_decisions_1');
%     filename=strcat('all_V_TF_1_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'all_V_TF_1');
%     cd(cd1)
    
%     
 
%% Retreinamento 
    case 4
     
%*************************************************************************
%    REMOVE COMMENTS TO SAVE DATA                                        *
%*************************************************************************

    cd1=cd;
    load('dV_threshold.mat');
    V_TF_up_to_40_mean=mean(all_V_TF_1(find(all_V_TF_1>40)));
    V_TF_award=V_TF_up_to_40_mean+(100-V_TF_up_to_40_mean)*dV_threshold;
    save('V_TF_award.mat','V_TF_award');
    cd('Database_BCI_Attention');
    cd(type_of_study);
    cd(intervation_or_sham_group);
    cd(Session_number);
    cd(Subject_ID);
    if  isdir('Phase_2_active_training')==0
        mkdir('Phase_2_active_training');
    end
    cd('Phase_2_active_training');
    filename=strcat('X_E_online_1_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'X_E_online_1');
    filename=strcat('X_E_online_1_all_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'X_E_online_1_all');
    filename=strcat('all_labels_1_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'all_labels_1');
    filename=strcat('all_attention_decisions_1_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'all_attention_decisions_1');
    filename=strcat('all_V_TF_1_',Subject_ID,'_M',num2str(Modo),'.mat');
    save(filename,'all_V_TF_1');
    cd(cd1)    
        
    clear
    clc

    cd1=cd;
    load('intervation_or_sham_group.mat')
    load('Session_number.mat');
    load('Subject_ID.mat');
    load('Modo.mat');
    load('type_of_study.mat')
    if Modo ==3
         load minimal_time_sustained_attention
    end
    cd('Database_BCI_Attention');
    cd(type_of_study);
    cd(intervation_or_sham_group);
    cd(Session_number);
    cd(Subject_ID);
    cd('Phase_2_active_training');
    filename=strcat('X_E_online_1_',Subject_ID,'_M',num2str(Modo),'.mat');
    load (filename) 
    cd(cd1)
    
    load Ctemp
    load dados_relaxado_treino
    load dados_relaxado_teste
    load dados_atencao_treino
    load dados_atencao_teste
    load media_desatencao   
    load Media_treino
    load Desvio_padrao_treino
    load theta_beta_ratio;
    
    dados_atencao = [permute(dados_atencao_treino,[1 3 2])  permute(X_E_online_1,[1 3 2])];
    dados_atencao = permute( dados_atencao,[1 3 2]); 
    Ctest_atencao = covariances(dados_atencao); 
    atencao_set = Tangent_space(Ctest_atencao,Ctemp)'; 
    atencao_set = (atencao_set - Media_treino)./Desvio_padrao_treino;
        
    Ctest_treino = covariances(dados_relaxado_treino); 
    desatencao_set_treino = Tangent_space(Ctest_treino,Ctemp)'; 
    desatencao_set_treino = (desatencao_set_treino - Media_treino)./Desvio_padrao_treino;
    
    A = atencao_set;% set de desatenção
    D = desatencao_set_treino;% set de desatenção treino
    E = [D;A];
    
    [nana,nanana,Na]=size(dados_atencao_treino);
    [N,NN]=size(A);
    [M,MM]=size(D);
    [K,KK]=size(E);
    clear dist_ij;
    clear P_erro;
    for i = 1:N
        for j = 1:K
            dist_ij(j,1) = sum((E(j,:) - A(i,:)).^2).^0.5;

        end 
        dist_ij(M+i,:)=[];
        Pij = exp(-dist_ij)./sum(exp(-dist_ij));
        P_erro(i) = sum(Pij(1:M,:));
    end
    
    P_erro_T1=P_erro(1:Na);
    P_erro_T2=P_erro(Na+1:end);
    P_erro_threshold=max(P_erro_T1)*(1-abs(max(P_erro_T1)-max(P_erro_T2)))
    true_attention_patterns = find(P_erro <= P_erro_threshold);
    
    dados_atencao_treino = dados_atencao(:,:,true_attention_patterns);

%% dados atenção e desatencao treino e teste.   
        
  %  dados_atencao_treino = dados_atencao(:,:,true_attention_patterns(1:end/2)) ;
  %  dados_atencao_teste =  dados_atencao(:,:,true_attention_patterns(end/2+1:end));
   
%% balanceando os dados de atencao e desatencao
    [qw,qwe,p]  = size(dados_atencao_treino);
    [as,asd,p1] = size(dados_atencao_teste);
    [zx,zxc,p2] = size(dados_relaxado_treino);
    [rt,rty,p3] = size(dados_relaxado_teste);
    

    
     if p>p2
        
        samples=randperm(p);
        dados_atencao_treino = dados_atencao_treino(:,:,samples(1:p2));
        p = p2;
      elseif p2>p
        
        samples=randperm(p2);
        dados_relaxado_treino = dados_relaxado_treino(:,:,samples(1:p));
        p2=p;
      end
    
%       if p1>p3
%         
%           samples=randperm(p1);
%           dados_atencao_teste = dados_atencao_teste(:,:,samples(1:p3));
%            p1=p3;
%       elseif p3>p1
%           
%           samples=randperm(p3);
%           dados_relaxado_teste = dados_relaxado_teste(:,:,samples(1:p1));
%           p3=p1; 
%       end
    

%%  Matrix a atencao desatenção treino e teste

     dados_treino = [permute(dados_atencao_treino,[1 3 2])  permute(dados_relaxado_treino,[1 3 2])];
     dados_treino = permute(dados_treino,[1 3 2]);
     
     dados_teste = [permute(dados_atencao_teste,[1 3 2])  permute(dados_relaxado_teste,[1 3 2])];
     dados_teste = permute(dados_teste,[1 3 2]);
    
     
    % Criando Ctemp e o vetor característico para treinar e validar
    Ctrain_AD = covariances(dados_treino);
    Ctemp= mean_covariances(Ctrain_AD,'riemann');
    full_training_set = Tangent_space(Ctrain_AD,Ctemp)'; 
    Media_treino = mean(full_training_set);
    Desvio_padrao_treino = std(full_training_set);
    full_training_set = (full_training_set - Media_treino)./Desvio_padrao_treino;
    
    % vetor característico para testar atencao
    Ctest = covariances(dados_teste);
    full_testing_set = Tangent_space(Ctest,Ctemp)';
    full_testing_set =(full_testing_set - Media_treino)./Desvio_padrao_treino;
 
    %%  criando o modelo SVM 

    clear class_treino;
    class_treino = ones(p,1); %giving the label 1 for attention patterns
    class_treino(p+1:p2+p,1) = -1; %giving the label -1 for attention patterns
    
    clear class_teste;
    class_teste = ones(p1,1); 
    class_teste(p1+1:p3+p1,1) = -1;

    SVMModel_treino = fitcsvm(full_training_set,class_treino,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');

    CVSVMModel_treino = crossval(SVMModel_treino);
    classLoss = kfoldLoss(CVSVMModel_treino);
    Acuracia_treino = 100 - classLoss*100

    [Group,groupgroup1] = predict(SVMModel_treino,full_testing_set);
    Acuracia_teste = sum(Group == class_teste) / length(class_teste) * 100
    
    plot_PCA(full_training_set,class_treino);
    plot_PCA(full_testing_set,class_teste);
    
      %% velocidade

    N=length(Group)/2;
    full_atencao = find(Group(1:N)==1);
    full_desatencao = N+find(Group(N+1:end)==-1);
    
    
%      EEG_labels_plot = {'F7','Fp1','Fp2','F8','F3','Fz',...
%                      'F4','C3','Cz','P8','P7','Pz',...
%                      'P4','T3','P3','O1','O2','C4',...
%                      'T4'};

  
   frontal_lobe=[1:7]; %F7,Fp1,Fp2,F8,F3,Fz,F4
   temporal_lobe=[14,19]; %T7,T8
   
   TB_A_T1_F=mean(theta_beta_ratio(full_atencao,frontal_lobe),2);
  
   TB_NA_T1_F=mean(theta_beta_ratio(full_desatencao,frontal_lobe),2);   
   
   TB_A_T1_T=mean(theta_beta_ratio(full_atencao,temporal_lobe),2);
  
   TB_NA_T1_T=mean(theta_beta_ratio(full_desatencao,temporal_lobe),2); 
   
   TB_A_T1_FT=mean(theta_beta_ratio(full_atencao,[frontal_lobe temporal_lobe]),2);
  
   TB_NA_T1_FT=mean(theta_beta_ratio(full_desatencao,[frontal_lobe temporal_lobe]),2);  
   
   TB_A_T1=[TB_A_T1_F TB_A_T1_T TB_A_T1_FT];
   TB_NA_T1=[TB_NA_T1_F TB_NA_T1_T TB_NA_T1_FT];
   
   
   [IIII,I]=max([abs(mean(TB_A_T1_F)-mean(TB_NA_T1_F)) abs(mean(TB_A_T1_T)-mean(TB_NA_T1_T)) abs(mean(TB_A_T1_FT)-mean(TB_NA_T1_FT))]);
   
    
    weights_attention=1*(2-Modo)/length(full_atencao)+(Modo-1)*(1./TB_A_T1(:,I))./sum((1./TB_A_T1(:,I)));
    media_atencao = sum(full_testing_set(full_atencao,:).*weights_attention);
    weights_non_attention=1*(2-Modo)/length(full_desatencao)+(Modo-1)*TB_NA_T1(:,I)./sum(TB_NA_T1(:,I));
    media_desatencao = sum(full_testing_set(full_desatencao,:).*weights_non_attention);
   
   % media_atencao = mean(full_testing_set(full_atencao,:));
   % media_desatencao = mean(full_testing_set(full_desatencao,:));

    % Distancia Euclidiana entre os centroids
    dist_euc = sum((media_desatencao - media_atencao).^2).^0.5;

    % Pegando os dados que foram classificados corretamente como atenção e
    % desatenção
    recognized_attention_set= full_testing_set(full_atencao,:);
    recognized_non_attention_set = full_testing_set(full_desatencao,:);

    % Calculando as distancias entre o centroid e todos os pontos da atenção
    clear dist_centroid_atencao;
    for i = 1:1:length(full_atencao)
       dist_centroid_atencao(i) = sum((media_atencao - recognized_attention_set(i,:) ).^2).^0.5;

    end

    % Calculando as distancias entre o centroid e todos os pontos da desatenção
    clear dist_centroid_desatencao;
    for i = 1:1:length(full_desatencao)
       dist_centroid_desatencao(i) = sum((media_desatencao - recognized_non_attention_set(i,:) ).^2).^0.5;

    end

    %Limitando os dados entre Lim_Inf e Lim_Sup
    Quant_val = quantile(dist_centroid_atencao,[0.25 0.50 0.75 1]);
    Lim_Sup = Quant_val(3) + 1.5*(Quant_val(3) - Quant_val(1));
    Lim_Inf = Quant_val(3) - 1.5*(Quant_val(3) - Quant_val(1));
    I = find(dist_centroid_atencao > Lim_Inf & dist_centroid_atencao < Lim_Sup);
    dist_centroid_atencao = dist_centroid_atencao(I);

    Quant_val_des = quantile(dist_centroid_desatencao,[0.25 0.50 0.75 1]);
    Lim_Sup_des = Quant_val_des(3) + 1.5*(Quant_val_des(3) - Quant_val_des(1));
    Lim_Inf_des = Quant_val_des(3) - 1.5*(Quant_val_des(3) - Quant_val_des(1));
    I_des = find(dist_centroid_desatencao > Lim_Inf_des & dist_centroid_desatencao < Lim_Sup_des);
    dist_centroid_desatencao = dist_centroid_desatencao(I_des);

    %% Valores das Velocidades
    V_min = 40;
    V_med = 70;
    V_max = 100;
    V_min_des = 0;
    V_med_des = 20;
    V_max_des = 40;

    V = interp1([Lim_Sup Quant_val(2) Lim_Inf],[V_min V_med V_max],dist_centroid_atencao,'cubic');
    P = polyfit(dist_centroid_atencao,V,3);
    V_TF = polyval(P,dist_centroid_atencao);

    V_des = interp1([Lim_Inf_des Quant_val_des(2) Lim_Sup_des],[V_min_des V_med_des V_max_des],dist_centroid_desatencao,'cubic');
    P_des = polyfit(dist_centroid_desatencao,V_des,3);
    V_TF_des = polyval(P_des,dist_centroid_desatencao);

    figure;
    plot(dist_centroid_atencao,V_TF,'ro')
    hold on;
    plot(dist_centroid_desatencao,V_TF_des,'bo')
    
    save('SVMModel_treino.mat','SVMModel_treino');
    save('Ctemp.mat','Ctemp')
    save('V_min.mat','V_min')
    save('V_max.mat','V_max')
    save('V_min_des.mat','V_min_des')
    save('V_max_des.mat','V_max_des')
    save('Lim_Sup.mat','Lim_Sup')
    save('Lim_Inf_des.mat','Lim_Inf_des')
    save('media_atencao.mat','media_atencao')
    save('media_desatencao.mat','media_desatencao')
    save('P.mat','P')
    save('P_des.mat','P_des')
    save('Media_treino.mat','Media_treino')
    save('Desvio_padrao_treino.mat','Desvio_padrao_treino')

%% Online 5 min
    case 5
        
    clear 
    close all
    clc
    
    cd1=cd;
    load('intervation_or_sham_group.mat');
    load('Subject_ID.mat');
    load('Modo.mat');
    load('Session_number.mat');
    load('V_TF_award.mat');
    load('histerese.mat');
    load('sham_mode.mat');
    load('type_of_study.mat');
 
    load SVMModel_treino
    load janela;
    load Ctemp
    load V_min
    load V_max
    load V_min_des
    load V_max_des
    load Lim_Sup
    load Lim_Inf_des
    load P
    load P_des
    load media_atencao
    load media_desatencao
    load Media_treino
    load Desvio_padrao_treino
    VH=V_max;
    VL=V_min;
    
    if Modo==3
        load VH
        load VL
        load minimal_time_sustained_attention 
        BCI_latency = 0;
    end
    
    eletrodos = [1:19];
    V_TF = 0;
    
    velocidade = 0;
    band = [1 30];
    vconst = 0;
    Group = -1*ones(5,1);
    group_true = -1;
    Fs = 500;
    k = 19;
    x = 0;
    x1=0;
    

    %% Comunicação com o unity
    client = Client(55001);

    %% instantiate the library
    disp('Loading the library...');
    lib = lsl_loadlib();

    % resolve a stream...
    disp('Resolving an EEG stream...');
    result = {};

    while isempty(result)
        result = lsl_resolve_byprop(lib,'type','EEG'); 
    end

    % create a new inlet
    disp('Opening an inlet...');
    inlet = lsl_inlet(result{1});

    disp('Now receiving data...');
    % Contador do vetor que vai comparar os 10 valores por segundo
    i = 1;

    % utilizado para salvar o matriz_ele_online e não sobrepor os valores mesmo
    % com o tamanho de chunk aleatorio
    aux = 0;

    %Contador para salvar energia
    contador = 1;

    %Parâmetros
    DATA_POINTS = janela;%1000;
    eletrodos_Total = 19;
    data_buffer = zeros(eletrodos_Total,DATA_POINTS);
    CHANNEL_OF_INTEREST = [1:19]; 
    SIZE_CHANNELS = size(CHANNEL_OF_INTEREST,2);

    while 1
        [vec,ts] = inlet.pull_chunk();    
        new_points = vec(1:eletrodos_Total,:);    
        new_length = size(new_points,2);

        % Guardar os valores dentro da matriz para posterior processamento
        matriz_ele_online(:,(aux+1):(new_length+aux)) = new_points;
        aux = new_length + aux;

        % Atualização do Buffer
        data_buffer(1:eletrodos_Total,1:DATA_POINTS-new_length) = data_buffer(1:eletrodos_Total,new_length+1:end);
        data_buffer(1:eletrodos_Total,DATA_POINTS-new_length+1:end) = new_points(1:eletrodos_Total,:);

    %% aux > quantidade de amostras por segundo , tbm mudar o DATA_POINTS para a quantidade de amostras    
        if(aux>janela)
            
          if x1==0 %& Modo==3  %First epoch
           tic;
          end

        X_E = fftfilterv2(data_buffer(eletrodos,:)',Fs,band)';

        Ctest = covariances(X_E); 
        validation_set = Tangent_space(Ctest,Ctemp)'; 
        validation_set = (validation_set - Media_treino)./Desvio_padrao_treino;

    % Classificação
        Group(1:end-1) = Group(2:end);
        [Groupp,groupgroup3] = predict(SVMModel_treino,validation_set);
        Group(end) = Groupp;
        
        

    %% Mandando a velocidade pro unity 
    if sum(Group==mode(Group))== length(Group) && mode(Group)==1
       group_true = 1;

    elseif sum(Group==mode(Group))== length(Group) && mode(Group)==-1
       group_true = -1;

    end
    
           %sham_mode=1 to facilite the flight speed control with very low
           %attention, 0 otherwise
           if group_true == 1 && Group(end)==1
              x = x + 1;
              X_E_online_atencao(:,:,x) = X_E;

             if Modo==3 
                 %The flight speed is controlled taking into account the sustained attention 
                velocidade = velocidade + group_true*(BCI_latency/minimal_time_sustained_attention)*100 + group_true*sham_mode*10;
             else
                %The flight speed is actively controlled
               dist_euc = sum((media_atencao - validation_set).^2).^0.5;
               velocidade = polyval(P,dist_euc)+sham_mode*40; %sham_mode=1 to facilite the flight speed control with very 
               VH=V_max;                                      %low attention
               VL=V_min;
             end
      
       
        elseif group_true == -1 && Group(end)==-1
           if Modo==3 
              %The flight speed is controlled taking into account the sustained attention   
              velocidade = velocidade + group_true*(BCI_latency/minimal_time_sustained_attention)*100 + group_true*sham_mode*10;
           else
              %The flight speed is actively controlled
              dist_euc = sum(( media_desatencao - validation_set).^2).^0.5;
              velocidade = polyval(P_des,dist_euc)+sham_mode*40; %sham_mode=1 to facilite the flight speed control with very 
              VH=V_max_des;                                      %low attention
              VL=V_min_des;
           end

        end
    
        
         %Comparing with histerese for maintaining maximum speed for a defined condition 
        if velocidade < VL 
           V_TF = VL;

        elseif (velocidade > VH | (Modo~=3 & histerese==1 & velocidade>=V_TF_award & V_TF == V_max) | (sham_mode==1 & velocidade>VL & V_TF == VH))
           V_TF = VH;
        else
           V_TF = velocidade 
        end


        client.send(V_TF)
        
         if x1==0 %& Modo==3  %First epoch
           BCI_latency = toc;
        end
        
        %Saving all data
        x1 = x1 + 1;
        X_E_online_full(:,:,x1) = X_E;
        full_labels_online(x1,1)=group_true;
        full_attention_decisions_online(x1,1)=(1+group_true)*(group_true*Group(end)+1)/4; %given 1 for attention decisions, and 0 otherwise
        full_V_TF_online(x1,1)=V_TF;

        

       end

    end  
end

%*************************************************************************
%    REMOVE COMMENTS TO SAVE DATA
%*************************************************************************

%     cd1=cd;
%     cd('Database_BCI_Attention');
%     cd(type_of_study);
%     cd(intervation_or_sham_group);
%     cd(Session_number);
%     cd(Subject_ID);
%     if  isdir('Phase_3_test_online')==0
%         mkdir('Phase_3_test_online');
%     end
%     cd('Phase_3_test_online');
%     filename=strcat('X_E_online_atencao_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'X_E_online_atencao');
%     filename=strcat('X_E_online_full_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'X_E_online_full');
%     filename=strcat('full_labels_online_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'full_labels_online');
%     filename=strcat('full_attention_decisions_online_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'full_attention_decisions_online');
%     filename=strcat('full_V_TF_online_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'full_V_TF_online');
%     filename=strcat('BCI_latency_',Subject_ID,'_M',num2str(Modo),'.mat');
%     save(filename,'BCI_latency');
%     cd(cd1);
