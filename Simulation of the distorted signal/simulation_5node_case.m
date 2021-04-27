clear;

N_symb=10;
xx=N_symb/5;
Ds= 10 ; %distance
SNR = 12 ;

rng('shuffle');
n=randi([0 1],1,N_symb);

signal_1=zeros(1,N_symb);
f_symb=1e10;

for ii=1:N_symb
    if n(ii)==0
        nn(ii)=0;
    else
        nn(ii)=1;
    end
end

a=1;
ts=0:(0.1)/(f_symb):(N_symb-0.1)/f_symb;

for ii=1:10*N_symb
    if  ii/(10*a)==1
        signal(ii)=nn(a);
        a=a+1;
    else
        signal(ii)=nn(a);
    end
end

%CD
c          = 2.99792458e8;     % Speed of light [m/s]
lambda     = 1.55e-6;          % Wavelength [m]
D_to_beta2 = -lambda^2/2/pi/c; % * D [s/m^2]



beta2_L    = D_to_beta2*17e-6*Ds*1e3;  
%17 means CD coefficient of -17 ps/(nm*km).

samp_per_symb=10;
bits_per_symb=1; %OOK
f_samp=f_symb*samp_per_symb; %Samplinh rate [Hz]
N_samp=N_symb*samp_per_symb; 
dt_symb=1/f_symb; % symbolinterval [s]
dt_samp=1/f_samp;
T_period = N_samp*dt_samp;
t        = 0:dt_samp:(N_samp - 1)*dt_samp;
f        =   1 /T_period*[(0:N_samp/2 - 1) (-N_samp/2:-1)];
omega    = 2*pi/T_period*[(0:N_samp/2 - 1) (-N_samp/2:-1)]; %omega vector 

for k = 1:size(signal)
      signal2(k,:) = ifft(fft(signal(k,:)).*exp(-1i.*1/2.*beta2_L.*omega.^2).*10^(-0.2*1/2*Ds));
end

rng('shuffle');

noise = randn(size(signal2))*std(signal2)/db2mag(SNR);
signal_noise=(signal2+ noise).^2;

subplot(411);
plot(t,signal,'m');
xlim([0 (N_symb+0.1)/(f_symb)]);
xlabel('Time(s)');
ylabel('Amplitude');
title('RZ Unipolar OOK signal');

subplot(412);
plot(t,(signal2).^2,'m');
xlim([0 (N_symb+0.1)/(f_symb)]);
xlabel('Time(s)');
ylabel('Amplitude');
title('the power of the recieved signal distorted only by CD and the loss in the fiber channel');

for abc=1:N_symb
   
    v=1;
    rn=10*(abc-3);
    
    if abc==1
        signal_noise_selected(1)=0;
        signal_noise_selected(2)=0;
        signal_noise_selected(3)=signal_noise(5);
        signal_noise_selected(4)=signal_noise(15);
        signal_noise_selected(5)=signal_noise(25);
        
        signal_selected5=signal(5);
        
    elseif abc==2
        signal_noise_selected(1)=0;
        signal_noise_selected(2)=signal_noise(5);
        signal_noise_selected(3)=signal_noise(15);
        signal_noise_selected(4)=signal_noise(25);
        signal_noise_selected(5)=signal_noise(35);
        
        signal_selected5=signal(15);
        
    elseif abc==N_symb-1
        signal_noise_selected(1)=signal_noise(5+(N_symb-4)*10);
        signal_noise_selected(2)=signal_noise(5+(N_symb-3)*10);
        signal_noise_selected(3)=signal_noise(5+(N_symb-2)*10);
        signal_noise_selected(4)=signal_noise(5+(N_symb-1)*10);
        signal_noise_selected(5)=0;
        
        signal_selected5=signal(5+(N_symb-2)*10);
        
    elseif abc==N_symb
        signal_noise_selected(1)=signal_noise(5+(N_symb-3)*10);
        signal_noise_selected(2)=signal_noise(5+(N_symb-2)*10);
        signal_noise_selected(3)=signal_noise(5+(N_symb-1)*10);
        signal_noise_selected(4)=0;
        signal_noise_selected(5)=0;
        
        signal_selected5=signal(5+(N_symb-1)*10);
        
    else
        for rr=1:5
    v=5;
    v=v+(rr-1)*10+rn;

    
    signal_noise_selected(rr,:)= signal_noise(v);
    
   
   
    if rr==3
        signal_selected5 = signal(v);
    end
    
        end
        
    end
   
  
    labeled(abc,:)=signal_selected5;
   
  
    output_real(abc,:)=[abs(signal_noise_selected(1)),abs(signal_noise_selected(2)),abs(signal_noise_selected(3)),abs(signal_noise_selected(4)),abs(signal_noise_selected(5))];

   
end


for abc=1:N_symb
   
    v=5;
    v=v+(abc-1)*10;


    signal_noise_selected5 = signal_noise(v);

    
    signal_noise_select(abc)= abs(signal_noise_selected5);
    

    
    end
  

subplot(413);
plot(t,signal_noise,'m');
xlim([0 (N_symb+0.1)/(f_symb)]);
xlabel('Time(s)');
ylabel('Amplitude');
title('the power of the recevied signal distorted by CD, loss and noise');


cHeader = { 'real' 'real' 'real' 'real' 'real'  'true' 'true' 'true' 'true' 'true' }; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas
%write header to file

fid = fopen('actual value vs true value.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);




MM=[ output_real labeled];
dlmwrite('actual value vs true value.csv',MM,'-append');
% dlmwrite('output_real.csv',output_real,'-append');

original=[];

threshold_value=mean(signal_noise_select)*0.5;

for xyz=1:N_symb
    
    reciever=signal_noise_select(xyz);   

    if reciever>=threshold_value
        original(xyz)=1;
    else
        original(xyz)=0;
    end
     
end

(N_symb-nnz(n-original))/N_symb
