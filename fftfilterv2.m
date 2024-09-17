function [Xf]= fftfilterv2(data,Fs,BW)

x = data;
L = size(x,1);
% NFFT=2^nextpow2(L);
NFFT = L;
f = Fs/2*linspace(0,1,NFFT/2+1);

% Applying FFT
Y = fft(x,NFFT);
N = size(Y,1);

if(BW(1)>0)
Y(1,:)=0;  %removing DC
end

%finding other components to remove
I = find(f(2:end)<BW(1) | f(2:end)>BW(2)); 

%removing components from one side
Y(1+I,:) = 0;  

%removing components from another side
Y(N+1-I,:) = 0; 

% Inverse FFT
Xf = ifft(Y,NFFT); %getting x filtered
Xf = Xf(1:length(data),:);

end

