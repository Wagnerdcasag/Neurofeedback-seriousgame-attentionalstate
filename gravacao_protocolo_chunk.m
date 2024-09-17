function [matriz_ele_online] = gravacao_protocolo_chunk()
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
   
        
     
     
    %% Saving all data
    %x1 = x1 + 1;
    %X_E_online_1_all(:,:,x1) = X_E;
    %all_labels_1(x1,1)=group_true;
    %all_attention_decisions_1(x1,1)=(1+group_true)*(group_true*Group(end)+1)/4; %given 1 for attention decisions, and 0 otherwise
    %all_V_TF_1(x1,1)=V_TF;
    
        
    end
end