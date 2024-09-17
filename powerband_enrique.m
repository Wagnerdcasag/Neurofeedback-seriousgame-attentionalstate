function [Pm] =  powerband_enrique(X,window,overlap,Fs,freq_range,mode)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if strcmp('pwelch',mode)==1
      [Pxx,F] = pwelch(X,window,overlap,[],Fs);
      
elseif strcmp('fft',mode)==1
    
     FFT_X_E  = fft(X);
     
     F = Fs/2*linspace(0,1,length(FFT_X_E)/2+1);
     %F = F(freq_range(1) & F<=freq_range(2));
     
     P2 = abs(FFT_X_E./size(FFT_X_E,1));
     P1 = P2(1:size(FFT_X_E,1)/2+1,:);
     P1(2:end-1,:) = 2*P1(2:end-1,:);
     Pxx = P1;
     
     Ip= find(F>=freq_range(1) & F<=freq_range(2));
     Pm=mean(Pxx(Ip,:));
     
elseif strcmp('Pband',mode)==1
         
     Pxx  = fftfilterv2(X,Fs,freq_range);
     %Pxx = Pxx./max(max(Pxx));
     %Pm = var(Pxx);%
     Pm=mean(Pxx.^2);
    
%       mu=mean(Pxx);
%       T=length(Pxx);
%       Pm=(sqrt(T)-sum(abs(Pxx-mu))/sqrt(sum((Pxx-mu).^2)))/(sqrt(T)-1);

     
end

      
      
end

