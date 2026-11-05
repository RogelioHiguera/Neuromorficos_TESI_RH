%Modelo de Izhikevich
%Paper Simple Model of Spiking Neurons 2003
%Ejemplo 2: Una sola neurona 
clc;clear;
Ne = 1;
Ni = 0; 
re = rand(Ne,1);
ri = rand(Ni,1);
a = [0.02*ones(Ne,1); 0.02+0.08*ri];
b = [0.2*ones(Ne,1); 0.25-0.05*ri];
c = [-65+15*re.^2; -65*ones(Ni,1)];
d = [8-6*re.^2; 2*ones(Ni,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = [0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v = -65*ones(Ne+Ni,1); % Valor inicial de v
u = b.*v; % Valor inicial de u
firings = []; % Instantes de disparo neuronal

for t = 1:1000 % Simulación de 1000ms      
    I = [5*randn(Ne,1);2*randn(Ni,1)]; % Entrada talámica
    fired = find(v>=30); % Indices de spikes 
    firings = [firings; t+0*fired, fired];
    v(fired) = c(fired);
    u(fired) = u(fired)+d(fired);
    I = I+sum(S(:,fired),2);
    v = v+0.5*(0.04*v.^2+5*v+140-u+I);
    v = v+0.5*(0.04*v.^2+5*v+140-u+I);
    u = u+a.*(b.*v-u);
    vplot(t) = v;
    uplot(t) = u;
end;
figure(1)
subplot(3,1,1)
plot(1:1000,vplot)
xlabel('Tiempo (ms)')
ylabel('v (mV)')
title('Potencial de membrana')
grid on
subplot(3,1,2)
plot(1:1000,uplot)
xlabel('Tiempo (ms)')
ylabel('u')
title('Variable de recuperación')
grid on
subplot(3,1,3)
plot(firings(:,1),firings(:,2),'.');
