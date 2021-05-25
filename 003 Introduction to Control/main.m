R_r = 0.075;
R_w = 0.03;
V_x = 0.15;
K_d = 2;
K_p = 1;
init = [0.1,0,0]';

[t,X] = ode45(@(t,X) system_fun(t,X), [0:0.001:50], init,'.');

y_r = 0.05*cos(t/pi);
dy_r = -0.05*(1/pi)*sin(t/pi);

w_z = (- K_d * (X(:,2) - dy_r) - K_p * (X(:,1) - y_r)) / V_x;
w_R = R_r * w_z / (2*R_w) + V_x / R_w;
w_L = 2 * V_x / R_w - w_R;

figure(1)
plot(t,X(:,1),t,y_r);
title('y, y_r')
legend('y','y_r');
xlabel('time(s)');
ylabel('y(m)')

figure(2)
plot(t,X(:,3));
title('e_{\psi}')
legend('e_{\psi}');
xlabel('time(s)');
ylabel('e_{\psi}(rad)')

figure(3)
plot(t,w_R,t,w_L);
title('w_R, w_L')
legend('w_R','w_L');
xlabel('time(s)');
ylabel('w(rad/s)')
