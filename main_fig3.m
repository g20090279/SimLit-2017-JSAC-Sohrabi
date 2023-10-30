% This File Is to Reproduce Fig. 3: SU-MIMO of the Paper
% - Sohrabi, F., & Yu, W. (2017). Hybrid analog and digital beamforming for 
% mmWave OFDM large-scale antenna arrays. *IEEE Journal on Selected Areas 
% in Communications, 35*(7), pp. 1432-1443. https://doi.org/10.1109/JSAC.2017.2698958
% Last Revised: August 7, 2023
% Zekai Liang, liang@ice.rwth-aachen.de
%

%% System Parameters
addpath("../communication-systems");
addpath("../KaiLib/KaiLib_matlab/");

numMC = 100;

snrDb = 20;
snr = 10^(snrDb/10);
numSubcarrier = 32;
numRf = 4;
numStream = numRf;
numUser = 32;
numAntTxVec = [16,32,64,128,256,512,800];
lenAntTx = length(numAntTxVec);

% Channel Settings
limAoA = [0,360]; % in degree
limAoD = [0,360]; % in degree
numClusterVec = [15,5]; % Two Types of Channel: (Nc,Nsc)=(15,1) & =(5,10)
numRayVec = [1,10];
numChnType = 2;

%% Initialize Variables
% The First Row is For Channel Type 1, and The Second Row For Type 2
capFd = zeros(lenAntTx,2);
capHb = zeros(lenAntTx,2);

for iMC = 1:numMC
    for iType = 1:numChnType
        for iAnt = 1:lenAntTx
            numAntTx = numAntTxVec(iAnt);
            numAntRx = numAntTx;
            numCluster = numClusterVec(iType);
            numRay = numRayVec(iType);

            %% Generate Channels
            Chn = ChannelModel( ...
                'ChannelType', 'mmWave', ...
                'NumSubcarrier', numSubcarrier, ...
                'NumAntTx', numAntTx, ...
                'NumAntRx', numAntRx, ...
                'NumCluster', numCluster, ...
                'NumRay', numRay, ...
                'AngularSpread', 10, ...
                'LimitAoD', limAoD, ...
                'LimitAoA', limAoA, ...
                'TapIndex', 0:numCluster-1 ...
                );
    
            % Put Frequently Used Object Variables in Local Variables
            chnFreq = Chn.ChnCoeffFreq;
    
            %% Fully-Digital Beamforming in MIMO OFDM
            capTmp = 0;
            for iSubc = 1:numSubcarrier
                chnTmp = chnFreq(:,:,iSubc);
                eigVal = eig(chnTmp'*chnTmp); % Note: eig()'s result is unsorted.
                gainChn = maxk(eigVal,numStream);
        
                % Note: FD beamformer diagonalizes the Channel. Therefore, the
                % logdet() of a diagonal channel becomes sum(log()) of the diagonal
                % elements
                capTmp = capTmp + sum( log2(1+1/numStream*gainChn*snr) ).';
            end
            capFd(iAnt,iType) = capFd(iAnt,iType) + 1/numSubcarrier*capTmp;
    
            %% Proposed Hybrid Beamforming in MIMO OFDM
            % Initialize Variables
            Vd = zeros(numRf,numStream,numSubcarrier);
            Vt = zeros(numAntTx,numStream,numSubcarrier);
            Wd = zeros(numRf,numStream,numSubcarrier);
            Wt = zeros(numAntRx,numStream,numSubcarrier);
    
            % TX Analog Part
            F1 = 0;
            for iSub = 1:numSubcarrier
                chnTmp = chnFreq(:,:,iSubc);
                F1 = F1 + chnTmp'*chnTmp;
            end
            F1 = F1 / numSubcarrier;
            [Vrf,~] = hbf_algorithm1(F1,snr,numRf);
            F2 = 0; % For RX Analog Part
            
            for iSubc = 1:numSubcarrier
                chnFreqTmp = chnFreq(:,:,iSubc);
        
                % Tx Digital Part, OFDM
                Q = Vrf'*Vrf;
                Heff = chnFreqTmp*Vrf;
                [~,~,V] = svd(Heff*Q^(-1/2));
                VdTmp = Q^(-1/2)*V(:,1:numStream)/sqrt(numStream); % norm(Vrf*Vd,'fro')=1
                Vd(:,:,iSubc) = VdTmp;
                
                VtTmp = Vrf*VdTmp;
                F2 = F2 + chnFreqTmp*(VtTmp*VtTmp')*chnFreqTmp';
                Vt(:,:,iSubc) = VtTmp;
            end
            
            % RX Analog Part
            F2 = snr * F2 / numSubcarrier;  % take the mean instead of sum
            [Wrf,~] = hbf_algorithm1(F2,1/numAntRx,numRf);
                
            % RX Digital Part
            for iSubc = 1:numSubcarrier
                chnFreqTmp = chnFreq(:,:,iSubc); % put it in a temporary variable
                VtTmp = Vt(:,:,iSubc);
                J = snr*Wrf'*(chnFreqTmp*(VtTmp*VtTmp')*chnFreqTmp')*Wrf + (Wrf'*Wrf);
                WdTmp = sqrt(snr)*J^(-1)*Wrf'*chnFreqTmp*VtTmp;
                Wd(:,:,iSubc) = WdTmp;
                tmp = Wrf*WdTmp;
                Wt(:,:,iSubc) = Wrf*WdTmp;
            end
        
            % Calcuate the Capacity
            capTmp = 0;
            for iSubc = 1:numSubcarrier
                Vtt = Vt(:,:,numSubcarrier);
                Wtt = Wt(:,:,numSubcarrier);
                H = chnFreq(:,:,iSubc);
                capTmp = capTmp + abs(log2(det( ...
                    eye(numAntRx) + snr*Wtt*(Wtt'*Wtt)^(-1)*Wtt'*(H*(Vtt*Vtt')*H') )));
            end
            capHb(iAnt,iType) = capHb(iAnt,iType) + 1/numSubcarrier*capTmp;

            %% Display the Progress
            waitingBar((iMC-1)*numChnType*lenAntTx+(iType-1)*lenAntTx+iAnt,...
                numMC*numChnType*lenAntTx);
        end
    end
end

% Average the Monte-Carlo Sum
capFd = capFd / numMC;
capHb = capHb / numMC;

%% Plot Figure
% Fig. 3 a
figure;
plot(numAntTxVec,capFd(:,1),'-+k',numAntTxVec,capHb(:,1),'-ob',LineWidth=1);
xlabel('Number of Antennas (N)');
ylabel('Spectral Efficiency bits/s/Hz')
title('(Nc,Nsc)=(15,1)')
lgd = legend('Optimal Full-digital BF','Asymptotic Hybrid BF');
lgd.Location = "northwest";
grid on;

% Fig. 3 b
figure;
plot(numAntTxVec,capFd(:,2),'-+k',numAntTxVec,capHb(:,2),'-ob',LineWidth=1);
xlabel('Number of Antennas (N)');
ylabel('Spectral Efficiency bits/s/Hz')
title('(Nc,Nsc)=(5,10)')
lgd = legend('Optimal Full-digital BF','Asymptotic Hybrid BF');
lgd.Location = "northwest";
grid on;