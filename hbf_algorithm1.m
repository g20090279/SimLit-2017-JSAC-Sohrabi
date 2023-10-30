function [V,C] = hbf_algorithm1(F,alpha,numColV,mode)
% HBF_ALGORITHM1 solves the following optimization problem by an iterative
% coordinate descent algorithm.
%
%          max_Vrf    C = log2(det( I + alpha*V'*F*V ))
%            s.t.      |V(i,j)|^2 <= 1, for all i, j
%
% where V is a numRowV-by-numColV matrix, F is a numColV-by-numColV matrix.
% alpha is a scalar containing SNR value in linear.
% The objective function can be rewritten to a simpler form by decomposing
% matrix F blockwise, i.e.
%
%                          | scalar,  | row vector |
%                     F =  | ---------|------------|
%                          |  column  |            |
%                          |  vector, |    matrix  |
%
% Note that V(i,j) can be zero. For partially-connected hybrid beamformer
% structure, V(i,j)=0 is the j-th RF chain is not connected to i-th
% antenna.
%
% Input(s):
% - F: a square matrix with size numRowV-by-numRowV.
% - alpha: a scalar.
% - numRowV: the number of rows for matrix V with size numRowV-by-numColV
%
% Output(s):
% - V: a matrix.
% - C: a scalar.

if nargin < 4, mode = "fully-connected"; end

% Make Sure F is A Square Matrix
numRowV = size(F);
if numRowV(1)~=numRowV(2), error('F should be a square matrix!'); end
numRowV = numRowV(1);

% For Partially-Connected Structure Only
numAntSub = numRowV / numColV;
if mod(numAntSub,1)~=0, error("The number of antennas in each subarray should be an integer"); end

% Convergence Threshold
epsilon = 10^-3;

% Initialize V
% Note: Initialization with All Ones Can Slightly Outperforms That with 
% Random Phase.
% V = exp(1i*2*pi*rand([numRowV,numColV]));
V = ones(numRowV,numColV);

% Calculate Initial Capacity
C = abs(log2(det( eye(numColV) + alpha*V'*F*V )));

% Initialize Variables
diff = inf;
count = 0;

while diff > epsilon
    for iCol = 1:numColV
        VrfBar = V;
        VrfBar(:,iCol) = [];
        Cj = eye(numColV-1) + alpha*VrfBar'*F*VrfBar;
        Gj = alpha*F - alpha^2*F*VrfBar*Cj^(-1)*VrfBar'*F;
        for iRow = 1:numRowV
            % Obtain Vector Gj(i,l) and Vrf(l,j) Where l~=i
            gj = Gj(iRow,:);
            gj(iRow) = [];
            vrfj = V(:,iCol);
            vrfj(iRow) = [];

            % Eq.(14) Determines Vrf(i,j) value
            if mode == "fully-connected"
                V(iRow,iCol) = exp(1i*angle(gj*vrfj));
            elseif mode == "partially-connected"
                if iRow >= (iCol-1)*numAntSub+1 && iRow <= iCol*numAntSub
                    V(iRow,iCol) = exp(1i*angle(gj*vrfj));
                else
                    V(iRow,iCol) = 0;
                end
            else
                error("Not supported mode.");
            end
        end
    end

    % Calculate the New Capacity, Assuming Vd*Vd'=I
    cNew = abs(log2(det( eye(numColV) + alpha*V'*F*V )));
    diff = abs(cNew-C)/C;
    C = cNew;

    % Count How Many Iterations to Converge
    count = count+1;
end

% Display Which Iteration We Are In
disp(['For alpha=',num2str(alpha),': converged after ',num2str(count),' iterations.']);
end