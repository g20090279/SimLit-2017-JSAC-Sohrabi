% This File Is to Reproduce Fig. 4: SU-MIMO of the Paper
% - Sohrabi, F., & Yu, W. (2017). Hybrid analog and digital beamforming for 
% mmWave OFDM large-scale antenna arrays. *IEEE Journal on Selected Areas 
% in Communications, 35*(7), pp. 1432-1443. https://doi.org/10.1109/JSAC.2017.2698958
% Last Revised: August7, 2023
% Zekai Liang, liang@ice.rwth-aachen.de

%% System Parameters
addpath("../communication-systems");
addpath("../KaiLib/KaiLib_matlab/");

% Channel Parameters
numCluster = 5;
numRay = 10;
limAoA = [0,360]; % in degree
limAoD = [0,360]; % in degree
angularSpread = 10; % in degree
numAntTx = 64;
numAntRx = 32;
numSubcarrier = 64;
numRf = 4;
numStream = 2;

% Monte-Carlo Simulation Parameter
numMC = 100;

% SNR Range
snrVecDb = (-15:5:10).';
snrVec = 10.^(snrVecDb/10);
snrLen = length(snrVec);

% Initial Variables
capFd = 0;   % capacity for fully-digital beamformer
capHb = 0; % 1st col for fully-connected, and 2nd one for partially-connected

%% Begin Monte-Carlo Simulation
for indMC = 1:numMC
    %% Generate Channels
    Chn = ChannelModel( ...
        'ChannelType', 'mmWave', ...
        'NumSubcarrier', numSubcarrier, ...
        'NumAntTx', numAntTx, ...
        'NumAntRx', numAntRx, ...
        'NumCluster', numCluster, ...
        'NumRay', numRay, ...
        'AngularSpread', angularSpread, ...
        'LimitAoD', limAoD, ...
        'LimitAoA', limAoA, ...
        'TapIndex', 0:numCluster-1 ...
        );
    
    % Put Frequently Used Object Variables in Local Variables
    chnFreq = Chn.ChnCoeffFreq;

    %% Fully-Digital Beamforming
    capTmp = 0;
    for iSubc = 1:numSubcarrier
        chnTmp = chnFreq(:,:,iSubc);
        eigVal = eig(chnTmp'*chnTmp); % Note: eig()'s result is unsorted.
        gainChn = maxk(eigVal,numStream);

        % Note: FD beamformer diagonalizes the Channel. Therefore, the
        % logdet() of a diagonal channel becomes sum(log()) of the diagonal
        % elements
        capTmp = capTmp + sum( log2(1+1/numStream*gainChn*snrVec.') ).';
    end
    capFd = capFd + 1/numSubcarrier*capTmp;

    %% Proposed Hybrid Beamforming with Fully-/Partially-Connected Structure
    F1 = 0;
    for iSub = 1:numSubcarrier
        chnTmp = chnFreq(:,:,iSubc);
        F1 = F1 + chnTmp'*chnTmp;
    end
    F1 = F1 / numSubcarrier;
    
    capTmp = zeros(snrLen,2);
    for iMode = 1:2 % Two Modes: Fully- / Partially- Connected Structure
        for snrInd = 1:snrLen
            % Initialize Variables
            Vd = zeros(numRf,numStream,numSubcarrier);
            Vt = zeros(numAntTx,numStream,numSubcarrier);
            Wd = zeros(numRf,numStream,numSubcarrier);
            Wt = zeros(numAntRx,numStream,numSubcarrier);
            F2 = 0; % For RX Analog Part
        
            % TX Analog Part
            if iMode == 1
                [Vrf,~] = hbf_algorithm1(F1,snrVec(snrInd),numRf);
            else
                [Vrf,~] = hbf_algorithm1(F1,snrVec(snrInd),numRf,"partially-connected");
            end
            
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
            F2 = F2 / numSubcarrier;  % take the mean instead of sum
            if iMode == 1
                [Wrf,~] = hbf_algorithm1(F2,snrVec(snrInd)/numAntRx,numRf);
            else
                [Wrf,~] = hbf_algorithm1(F2,snrVec(snrInd)/numAntRx*numRf,numRf,"partially-connected");
            end
                
            % RX Digital Part
            for iSubc = 1:numSubcarrier
                chnFreqTmp = chnFreq(:,:,iSubc); % put it in a temporary variable
                VtTmp = Vt(:,:,iSubc);
                J = snrVec(snrInd)*Wrf'*(chnFreqTmp*(VtTmp*VtTmp')*chnFreqTmp')*Wrf + (Wrf'*Wrf);
                WdTmp = sqrt(snrVec(snrInd))*J^(-1)*Wrf'*chnFreqTmp*VtTmp;
                Wd(:,:,iSubc) = WdTmp;
                tmp = Wrf*WdTmp;
                Wt(:,:,iSubc) = Wrf*WdTmp;
            end
        
            % Calcuate the Capacity OFDM
            for iSubc = 1:numSubcarrier
                Vtt = Vt(:,:,numSubcarrier);
                Wtt = Wt(:,:,numSubcarrier);
                H = chnFreq(:,:,iSubc);
                capTmp(snrInd,iMode) = capTmp(snrInd,iMode) + abs(log2(det( ...
                    eye(numAntRx) + snrVec(snrInd)*Wtt*(Wtt'*Wtt)^(-1)*Wtt'*(H*(Vtt*Vtt')*H') )));
            end
        end
    end
    
    % Save the Capacity
    capHb = capHb + 1/numSubcarrier*capTmp;

    %% Asymptotic Hybrid BF (Nrf=Ns, Nt->oo, Nr->oo)
    numRf2 = numStream;
    numAntTx = 800;
    numAntRx = 800;


    %% Display the Progress
    waitingBar(indMC,numMC)
end

% average to obtain final capacity
capFd = capFd / numMC;
capHb = capHb / numMC;

% plot figure
figure();
grid on; hold on;
plot(snrVecDb,capFd,'k-',snrVecDb,capHb(:,1),'b-o',snrVecDb,capHb(:,2),'b--o');
xlabel('SNR (dB)');
ylabel('Spectral Efficiency (bits/s/Hz)')
lgd = legend('Optimal Fully-digital Beamforming','Proposed Hybrid BF (fully-connected)',...
    'Proposed Hybrid BF (partially-connected)');
lgd.Location = "northwest";