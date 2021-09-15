 % This function calculates PPD for each seen and surrogate classes
% Inputs: 
%   training data:      X, Y
%   Genus labels:       G
%   Hyperparameters:    
%                       mu0     -- initial mean
%                       m       -- 
%                       k0      -- kappa_0
%                       k1      -- kappa_1
%                       Psi     -- Iinitial covariance matrix
%
% Outputs:
%   Class predictive covariances:  Sig_s
%   Class predictive means:        mu_s
%   Class predictive DoF:          v_s
%   Class ids:                     class_id

function [Sig_s,mu_s,v_s,class_id,Sigmas]=ppd_derivation(X,Y,G,Psi,mu0,m,k0,k1)

% data stats
seenclasses   = unique(Y);
seengenera = G(seenclasses);
uyg = unique(G);
ct = length(uyg);
nc            = length(seenclasses)+ct;
[n, d]        = size(X);

% Initialize output params: derive for each class predictive cov, mean and dof 
Sig_s         = zeros(d,d,nc);
Sigmas        = zeros(d,d,nc);
mu_s          = zeros(nc,d);
v_s           = zeros(nc,1);


% Starting with Seen classes
uy              = seenclasses;
ncl             = length(uy);
cnt           = 1; 

for i=1:ncl
    in          = Y==uy(i);
    Xi          = X(in,:);
    
    % The current selected component stats: # points, mean and scatter
    cur_n       = sum(in);
    cur_S       = (cur_n-1)*cov(Xi);
    cur_mu      = mean(Xi,1);
    
    % DATA likelihood and global prior. Note that we did not use local
    % priors for seen class PPD
    v_s(cnt)        = cur_n+m-d+1;
    mu_s(cnt,:)     = (cur_n*cur_mu+(k0*k1/(k0+k1))*mu0)/(cur_n+(k0*k1/(k0+k1)));
    Smu             = ((cur_n*(k0*k1/(k0+k1)))/((k0*k1/(k0+k1))+cur_n))*((cur_mu-mu0)*(cur_mu-mu0)');
    Sig_s(:,:,cnt)  = (Psi+cur_S+Smu)/(((cur_n+(k0*k1/(k0+k1)))*v_s(cnt))/(cur_n+(k0*k1/(k0+k1))+1));
    class_id(cnt,1) = uy(i);
    cnt             = cnt+1;    
end

% Now surrogate class PPD
uy            = unique(G);
ncl           = length(uy);

for i=1:ncl
    
    in = false(n,1);
    
    genus = uy(i);
    classes   =  seenclasses(seengenera==genus);
    nci       = length(classes);
    
    if (nci >= 1)
        for j=1:nci
            in(Y==classes(j))=1;
        end

        % Extract species/ classes belonging to this genus
        Yi        = Y(in);
        Xi        = X(in,:);
        uyi       = unique(Yi);

        % Initialize component sufficient statistics 
        ncpi      = length(uyi);
        xkl       = zeros(ncpi,d);      % Component means
        Skl       = zeros(d,d,ncpi);    % Component scatter matrices
        kap       = zeros(ncpi,1);      % model specific
        nkl       = zeros(ncpi,1);      % #data points in the components

        % Calculate  sufficient statistics for each component in meta cluster
        for j=1:ncpi
            in         = Yi==uyi(j);
            nkl(j)     = sum(in);
            kap(j)     = nkl(j)*k1/(nkl(j)+k1);
            Xij        = Xi(in,:);
            xkl(j,:)   = mean(Xij,1);
            Skl(:,:,j) = (nkl(j)-1)*cov(Xij);   
        end

        % Model specific parameters
        sumkap       = sum(kap);
        kaps         = (sumkap+k0)*k1/(sumkap+k0+k1);
        sumSkl       = sum(Skl,3);                                          % sum of scatters
        muk          = (sum(xkl.*(kap*ones(1,d)),1)+k0*mu0)/(sum(kap)+k0);  % meta cluster mean

        % Surrogate genus classes' predictive cov, mean and dof
        vsc             = sum(nkl)-ncpi+m-d+1;
        class_id(cnt,:) = uy(i);
        v_s(cnt)        = vsc;
        Sigmas(:,:,cnt) = Psi+sumSkl;
        Sig_s(:,:,cnt)  = (Psi+sumSkl)/(((kaps)*v_s(cnt))/(kaps+1));
        mu_s(cnt,:)     = muk;
        cnt             = cnt+1;
    end
end

 