function [out] = sim_noisyKF_cfrule_fast(cfg)

% check input arguments
if ~isfield(cfg,'rt')
    error('Missing experiment information!');
end
if ~all(isfield(cfg,{'alpha','zeta','tau','cfrule','nstype','chrule'}))
    error('Missing model parameters!');
end
if ~isfield(cfg,'r1st_rand')
    cfg.r1st_rand = true;
end
if ~isfield(cfg,'delta')
    if cfg.cfrule
        cfg.delta = nan;
    else
        error('Missing model parameters!');
    end
end

% get experiment information
rt     = cfg.rt;     % available rewards (nb x nt x 2)
nb     = size(rt,1); % number of blocks
nt     = size(rt,2); % number of trials

% define reparameterization functions:
%   * alpha = fa(vd/vs)
%   * vd/vs = fv(alpha)
fa = @(v)1./(1+exp(+0.4486-log2(v)*0.6282)).^0.5057;
av = 0.001:0.001:0.999;
vv = arrayfun(@(a)fzero(@(v)fa(v)-a,2.^[-30,+30]),av);
fv = @(a)interp1(av,vv,a,'pchip');

% set fixed statistics
m0 = 0.5000; % prior mean
v0 = 0.0214; %0.0455 (old); % prior variance
vs = 0.0163; %0.0236 (old); % sampling variance

% get model parameters
alpha  = cfg.alpha;  % learning rate
delta  = cfg.delta;  % decay rate
zeta   = cfg.zeta;   % learning noise
tau    = cfg.tau;    % policy temperature
cfrule = cfg.cfrule; % counterfactual rule
nstype = cfg.nstype; % noise type
chrule = cfg.chrule; % choice sampling rule

% clip parameter values to avoid numerical instability
alpha = min(max(alpha,0.001),0.999);
tau   = max(tau,1e-12);

% get drift variance
vd = fv(alpha)*vs;

% set first response flag
r1st_rand = cfg.r1st_rand;

% output variables
resp = nan(nb,nt);   % response
rgrd = nan(nb,nt);   % response greediness
rcur = nan(nb,nt);   % response curiosity
r_ch = nan(nb,nt);   % obtained rewards
r_un = nan(nb,nt);   % foregone rewards
ut   = nan(nb,nt,2); % unfiltered posterior means
mt   = nan(nb,nt,2); % posterior means
vt   = nan(nb,nt,2); % posterior variances
llr  = nan(nb,nt);   % log-likelihood ratio
lvr  = nan(nb,nt);   % log-variance ratio

% initialize posterior means and variances
mt(:,1,:) = m0;
vt(:,1,:) = v0;

% apply first response of each block
if r1st_rand
    % random
    resp(:,1) = ceil(2*rand(nb,1));
else
    % option 1 by default
    resp(:,1) = 1;
end

% get chosen and unchosen rewards
r_ch(:,1) = rt(sub2ind(size(rt),(1:nb)',ones(nb,1),resp(:,1)));
r_un(:,1) = rt(sub2ind(size(rt),(1:nb)',ones(nb,1),3-resp(:,1)));

for it = 2:nt
    
    % compute Kalman gain
    kgain = vt(:,it-1,:)./(vt(:,it-1,:)+vs);
    
    % update posterior means and variances

    % get blocks where chosen option = 1
    i1 = resp(:,it-1) == 1 & ~isnan(rt(:,it-1,1));
    
    % update posterior mean and variance of chosen option
    mt(i1,it,1) = mt(i1,it-1,1)+kgain(i1,1,1).*(rt(i1,it-1,1)-mt(i1,it-1,1));
    ut(i1,it,1) = mt(i1,it,1);
    if strcmp(nstype,'weber')
        mt(i1,it,1) = noisrnd(mt(i1,it,1), ...
            zeta*kgain(i1,1,1).*abs(rt(i1,it-1,1)-mt(i1,it-1,1)));
    else
        mt(i1,it,1) = noisrnd(mt(i1,it,1),zeta);
    end
    vt(i1,it,1) = (1-kgain(i1,1,1)).*vt(i1,it-1,1);
    
    if ~cfrule
        % decay posterior mean of unchosen option
        mt(i1,it,2) = mt(i1,it-1,2)+delta*(0.5-mt(i1,it-1,2));
        ut(i1,it,2) = mt(i1,it,2);
        vt(i1,it,2) = vt(i1,it-1,2);
    else
        % update posterior mean of unchosen option using counterfactual rule
        mt(i1,it,2) = mt(i1,it-1,2)+kgain(i1,1,2).*(1-rt(i1,it-1,1)-mt(i1,it-1,2));
        ut(i1,it,2) = mt(i1,it,2);
        if strcmp(nstype,'weber')
            mt(i1,it,2) = noisrnd(mt(i1,it,2), ...
                zeta*kgain(i1,1,2).*abs(1-rt(i1,it-1,1)-mt(i1,it-1,2)));
        else
            mt(i1,it,2) = noisrnd(mt(i1,it,2),zeta);
        end
        vt(i1,it,2) = (1-kgain(i1,1,2)).*vt(i1,it-1,2);
    end
    
    % get blocks where chosen option = 2
    i2 = resp(:,it-1) == 2 & ~isnan(rt(:,it-1,1));
    
    % update posterior mean and variance of chosen option
    mt(i2,it,2) = mt(i2,it-1,2)+kgain(i2,1,2).*(rt(i2,it-1,2)-mt(i2,it-1,2));
    ut(i2,it,2) = mt(i2,it,2);
    if strcmp(nstype,'weber')
        mt(i2,it,2) = noisrnd(mt(i2,it,2), ...
            zeta*kgain(i2,1,2).*abs(rt(i2,it-1,2)-mt(i2,it-1,2)));
    else
        mt(i2,it,2) = noisrnd(mt(i2,it,2),zeta);
    end
    vt(i2,it,2) = (1-kgain(i2,1,2)).*vt(i2,it-1,2);
    
    if ~cfrule
        % decay posterior mean of unchosen option
        mt(i2,it,1) = mt(i2,it-1,1)+delta*(0.5-mt(i2,it-1,1));
        ut(i2,it,1) = mt(i2,it,1);
        vt(i2,it,1) = vt(i2,it-1,1);
    else
        % update posterior mean of unchosen option using counterfactual rule
        mt(i2,it,1) = mt(i2,it-1,1)+kgain(i2,1,1).*(1-rt(i2,it-1,2)-mt(i2,it-1,1));
        ut(i2,it,1) = mt(i2,it,1);
        if strcmp(nstype,'weber')
            mt(i2,it,1) = noisrnd(mt(i2,it,1), ...
                zeta*kgain(i2,1,1).*abs(1-rt(i2,it-1,2)-mt(i2,it-1,1)));
        else
            mt(i2,it,1) = noisrnd(mt(i2,it,1),zeta);
        end
        vt(i2,it,1) = (1-kgain(i2,1,1)).*vt(i2,it-1,1);
    end
    
    % account for drift
    vt(:,it,:) = vt(:,it,:)+vd;
    
    % compute log-likelihood and log-variance ratios
    llr(:,it) = 2*(mt(:,it,1)-mt(:,it,2))./sqrt(vt(:,it,1)+vt(:,it,2));
    lvr(:,it) = log(vt(:,it,1))-log(vt(:,it,2));
    
    % apply policy to choose response
    if strcmp(chrule,'thomp')
        x = llr(:,it)/tau;
    else
        x = (mt(:,it,1)-mt(:,it,2))/tau;
    end
    resp(:,it) = 1+(rand(nb,1) > 1./(1+exp(-x)));
    
    % get response properties
    rgrd(:,it) = resp(:,it) == 1+(llr(:,it) < 0); % greediness
    rcur(:,it) = resp(:,it) == 1+(lvr(:,it) < 0); % curiosity
    
    % get chosen and unchosen rewards
    r_ch(:,it) = rt(sub2ind(size(rt),(1:nb)',it(ones(nb,1)),resp(:,it)));
    r_un(:,it) = rt(sub2ind(size(rt),(1:nb)',it(ones(nb,1)),3-resp(:,it)));
    
end

% remove respones in response matrix where there was no trial
resp(isnan(rt(:,:,1))) = nan;



% create output structure
out       = [];
out.m0    = m0;    % prior mean
out.v0    = v0;    % prior variance
out.vs    = vs;    % sampling variance
out.vd    = vd;    % drift variance
out.alpha = alpha; % learning rate
out.delta = delta; % learning decay
out.zeta  = zeta;  % learning noise
out.tau   = tau;   % policy temperature
out.resp  = resp;  % responses
out.rgrd  = rgrd;  % response greediness
out.rcur  = rcur;  % response curiosity
out.r_ch  = r_ch;  % obtained rewards
out.r_un  = r_un;  % foregone rewards
out.ut    = ut;    % unfiltered posterior means
out.mt    = mt;    % posterior means
out.vt    = vt;    % posterior variances
out.llr   = llr;   % log-likelihood ratio
out.lvr   = lvr;   % log-variance ratio

% use only when simulating subjects
out.resp_subj = cfg.resp_subj; % hold subject responses to compare in post
out.rond      = cfg.rond;

end