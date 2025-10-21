function [x_kf, varargout] = SquareRootKalmanFilter_CA(z, MAlen, varargin)
% SquareRootKalmanFilter_CA
% Extended to estimate 2D position, velocity, and acceleration
%
% Inputs:
%   z      - 2 x n noisy measurements of position [x; y]
%   MAlen  - length of moving average (for covariance estimation)
%   varargin - as before for N, 'EWMA', etc.
%
% Outputs:
%   x_kf   - 6 x n state estimates: [x; y; vx; vy; ax; ay]
%   varargout - same as original, with covariance info for 6 states

% Parse inputs (same as original)
if (nargin == 3)
    if ~ischar(varargin{1})
        N = varargin{1};
        MAType = 'EWMA';
        CovMethod = 'Standard';
    else
        N = MAlen;
        MAType = 'EWMA';
        CovMethod = 'EWMA';
    end
elseif (nargin == 4)
    N = varargin{1};
    MAType = varargin{2};
    CovMethod = 'Standard';
else
    error('Input syntax error. Type ''help SquareRootKalmanFilter_CA'' for assistance.');
end

%----------------------------------------------------------------------
% Dimensions and state definition
%----------------------------------------------------------------------

dt = 1;  % sampling interval, adjust if known

meas_dim = size(z,1); % Should be 2 for [x; y]
n = size(z,2);        % number of samples

state_dim = 6;        % [x; y; vx; vy; ax; ay]

% System matrices for constant acceleration model
A = [1 0 dt 0 0.5*dt^2 0;
     0 1 0 dt 0 0.5*dt^2;
     0 0 1 0 dt 0;
     0 0 0 1 0 dt;
     0 0 0 0 1 0;
     0 0 0 0 0 1];

H = [1 0 0 0 0 0;
     0 1 0 0 0 0];

%----------------------------------------------------------------------
% Initialize state estimate and covariances
%----------------------------------------------------------------------

x_apriori = zeros(state_dim,n);
x_aposteriori = zeros(state_dim,n);

UP_apriori = zeros(state_dim,state_dim,n);
DP_apriori = zeros(state_dim,state_dim,n);

UP_aposteriori = zeros(state_dim,state_dim,n);
DP_aposteriori = zeros(state_dim,state_dim,n);

Q = zeros(state_dim,state_dim,n);
UQ = zeros(state_dim,state_dim,n);
DQ = zeros(state_dim,state_dim,n);

R = zeros(meas_dim,meas_dim,n);
UR = zeros(meas_dim,meas_dim,n);
DR = zeros(meas_dim,meas_dim,n);

%----------------------------------------------------------------------
% Smooth measurements for initial state and covariance estimation
%----------------------------------------------------------------------

eval(['smoothed_z = ' MAType '(z,MAlen);']);

startIndex = N + MAlen - 1;

% Initial state guess: position from smoothed measurement, velocity & accel zero
x_aposteriori(:, startIndex-1) = [smoothed_z(:, startIndex-1); 0; 0; 0; 0];

% Initial covariance (identity)
UP_aposteriori(:,:,startIndex-1) = eye(state_dim);
DP_aposteriori(:,:,startIndex-1) = eye(state_dim);

%----------------------------------------------------------------------
% Now perform filtering loop
%----------------------------------------------------------------------

for i = startIndex:n
    % Estimate measurement noise and process noise covariances (on measurements)
    if strcmpi(CovMethod,'Standard')
        [R(:,:,i), Q_meas] = StandardCovEst(z, smoothed_z, i, N);
    else
        [R(:,:,i), Q_meas] = EWMACovEst(z, smoothed_z, i, N, R(:,:,i-1), Q(:,:,i-1));
    end

    % We must embed measurement noise covariance into state space size
    % So, R is meas_dim x meas_dim, Q must be state_dim x state_dim
    % Here we build Q based on process noise model (can tune process noise)
    
    % For example, assume Q_meas is measurement noise covariance (2x2),
    % we expand to 6x6 Q by adding process noise on velocity and acceleration
    % This is an approximation and can be tuned to your data
    
    q_pos = diag(diag(Q_meas));          % position noise (2x2)
    q_vel = eye(2) * 0.01;               % small velocity noise
    q_acc = eye(2) * 0.001;              % very small accel noise
    
    Q(:,:,i) = blkdiag(q_pos, q_vel, q_acc);

    try
        % Square root decompositions
        [UQ(:,:,i), DQ(:,:,i)] = myUD(Q(:,:,i));
        [UR(:,:,i), DR(:,:,i)] = myUD(R(:,:,i));
        
        % --- Time update (prediction) ---
        % Use your "thornton" function for time update, modified for new A matrix
        % Update signature should be (x_prev, UP_prev, DP_prev, UQ, DQ, A)
        % If your 'thornton' doesn't support A, you must update here manually:

        % Manual prediction:
        x_priori = A * x_aposteriori(:,i-1);
        P_aposteriori = UP_aposteriori(:,:,i-1) * DP_aposteriori(:,:,i-1) * UP_aposteriori(:,:,i-1)';
        P_priori = A * P_aposteriori * A' + Q(:,:,i);

        % Compute UD decomposition of P_priori (for square root filter)
        [UP_priori, DP_priori] = myUD(P_priori);

        x_apriori(:,i) = x_priori;
        UP_apriori(:,:,i) = UP_priori;
        DP_apriori(:,:,i) = DP_priori;

        % --- Measurement update ---
        % We have measurement z(:,i) (2x1)
        % Measurement matrix H (2x6)

        % Decorrelate measurements
        z_ind = myUnitTriSysSol(UR(:,:,i), z(:,i), 'upper');
        H_ind = myUnitTriSysSol(UR(:,:,i), H, 'upper');

        x_post = x_priori;
        UP_post = UP_priori;
        DP_post = DP_priori;

        for j = 1:meas_dim
            [x_post, UP_post, DP_post] = bierman(z_ind(j), DR(j,j,i), H_ind(j,:), x_post, UP_post, DP_post);
        end

        x_aposteriori(:,i) = x_post;
        UP_aposteriori(:,:,i) = UP_post;
        DP_aposteriori(:,:,i) = DP_post;

    catch e
        warning('Filter iteration failed at %d: %s', i, e.message);
        % Fallback: use measurement for position, zero velocity/accel
        x_aposteriori(:,i) = [z(:,i); 0; 0; 0; 0];
        UP_aposteriori(:,:,i) = eye(state_dim);
        DP_aposteriori(:,:,i) = eye(state_dim);
    end
end

% Return filtered state estimates
x_kf = x_aposteriori;

if nargout == 2
    varargout{1} = struct('x_pr', x_apriori, ...
                         'UP_pr', UP_apriori, ...
                         'DP_pr', DP_apriori, ...
                         'UP_po', UP_aposteriori, ...
                         'DP_po', DP_aposteriori, ...
                         'Q', Q, ...
                         'UQ', UQ, ...
                         'DQ', DQ, ...
                         'R', R, ...
                         'UR', UR, ...
                         'DR', DR);
end

end
