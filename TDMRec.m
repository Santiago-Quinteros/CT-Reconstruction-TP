N = 128 ;
phi = [0 : 1 : 180] ;
I0 = 1000 ;
tau = 1 ;
xtrue = phantom(N) ;
xtrue(xtrue<0) = 0 ;
xtrue = xtrue/10 ;
imagesc(xtrue) ;
axis image ;

sino = radon(xtrue, phi) ;
imagesc(sino);
axis image;

x1 = iradon(sino, phi, 'Linear', 'None', 1, N) ;
x2 = iradon(sino, phi, 'Linear', 'Ram-Lak', 1, N) ;
x3 = iradon(sino, phi, 'Linear', 'Hann', 1, N) ;

imagesc(x1);
axis image;

imagesc(x2);
axis image;

imagesc(x3);
axis image;

ybar = tau*I0*exp(-sino) ; % formule de Beer-Lambert
y = poissrnd(ybar) ;

imagesc(y);
axis image;

imagesc(ybar);
axis image;

clear xtrue ;
clear sino ;
clear ybar ;

b = log(tau*I0./y) ;
b(b==inf) = 0 ; % pour retirer les valeurs infinies

x_RamLak = iradon(b, phi, 'Linear', 'Ram-Lak', 1, N);
imagesc(x_RamLak) ;
axis image;

x_Hann = iradon(b, phi, 'Linear', 'Hann', 1, N);
imagesc(x_Hann) ;
axis image;

x = zeros(N,N) ;
alpha = 0.0001
w = y ;
Niter = 30;
loss_grad = zeros(1,Niter) ;
for i = 1:Niter
    grad_f = iradon(w.*(radon(x,phi)-b), phi, 'Linear', 'None', 1, N) ;
    x = x - alpha * grad_f
    loss_grad(i) = 0.5*sum(sum(w.*(radon(x,phi)-b).^2)) ;
    %imagesc(x);
    %axis image;
    %pause(0.1);
end

x = zeros(N,N) ;
w = y ;
loss = zeros(1,Niter) ;
for i = 1:Niter
    grad_f = iradon(w.*(radon(x,phi)-b), phi, 'Linear', 'None', 1, N) ;
    D = iradon(w.*radon(ones(N,N),phi), phi, 'Linear', 'None', 1, N) ;
    x = x - grad_f./D
    loss(i) = 0.5*sum(sum(w.*(radon(x,phi)-b).^2)) ;
    %imagesc(x);
    %axis image;
    %pause(0.1);
end

%Q.13
x_nest = zeros(N,N) ;
w = y ;
t = 1 ;
z = 0 ;
z_old = z ;
t_old = t ;
loss_nest = zeros(1,Niter) ;
for i = 1:10
    grad_f = iradon(w.*(radon(x_nest,phi)-b), phi, 'Linear', 'None', 1, N) ;
    D = iradon(w.*radon(ones(N,N),phi), phi, 'Linear', 'None', 1, N) ;
    z = x_nest - grad_f./D ;
    t = 0.5*(1 + sqrt(1 + 4*t^2));
    x_nest = z + (t_old - 1)/t*(z - z_old);
    t_old = t;
    z_old = z;
    loss_nest(i) = 0.5*sum(sum(w.*(radon(x_nest,phi)-b).^2)) ;
    %imagesc(x_nest);
    %axis image;
    %pause(0.1);
end

plot(1:Niter,loss,'red-');
hold on;
plot(1:Niter,loss_nest,'blue-');
hold on;
plot(1:Niter,loss_grad,'green-');
legend('TO','TO+Nest', 'grad desc');

param.dimImage = N ;
param.phi = phi ;
param.I0 = I0 ;
param.prior = 'quadratic' ;
param.beta_prior = 1000 ;
param.n_iter = 100 ;

x = recoCT_sps(y, param) ;
imagesc(x);
axis image;

param.dimImage = N ;
param.phi = phi ;
param.I0 = I0 ;
param.prior = 'Huber' ;
param.delta_huber = 0.001 ;
param.beta_prior = 1000 ;
param.n_iter = 100 ;

x = recoCT_sps(y, param) ;
imagesc(x);
axis image;

param.dimImage = N ;
param.phi = phi ;
param.I0 = I0 ;
param.beta_prior = 0.5 ;
param.rho_init = 10000 ;
param.prior = 'TVl1' ;
param.n_iter = 100 ;
param.n_inner_iter = 10 ;

x = recoCT_admm(y, param) ;
imagesc(x);
axis image;

phi = [0 : 15 : 180] ;
sino = radon(xtrue, phi) ;

x1 = iradon(sino, phi, 'Linear', 'None', 1, N) ;
x2 = iradon(sino, phi, 'Linear', 'Ram-Lak', 1, N) ;
x3 = iradon(sino, phi, 'Linear', 'Hann', 1, N) ;

imagesc(x1);
axis image;

imagesc(x3);
axis image;

imagesc(x3);
axis image;

ybar = tau*I0*exp(-sino) ; % formule de Beer-Lambert
y = ybar

param.dimImage = N ;
param.phi = phi ;
param.I0 = I0 ;
param.prior = 'Huber' ;
param.beta_prior = 300 ;
param.n_iter = 100 ;

x = recoCT_sps(y, param) ;
imagesc(x);
axis image;

param.dimImage = N ;
param.phi = phi ;
param.I0 = I0 ;
param.beta_prior = 0.5 ;
param.rho_init = 10000 ;
param.prior = 'TVl1' ;
param.n_iter = 100 ;
param.n_inner_iter = 10 ;

x = recoCT_admm(y, param) ;
imagesc(x);
axis image;