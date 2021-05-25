function dX = system_fun(t,X)
% %  input -> y, ODE -> e_y  
    y_r = 0.05 * cos(t/pi);
    dy_r = - 0.05*(1/pi)*sin(t/pi);
    ddy_r = - 0.05 * cos(t/pi) / pi / pi;
    
    dX = zeros(3,1);
    x = X(1);
    dx = X(2);
    e_psi = X(3);
    
    V_x = 0.15;
    K_d = 2;
    K_p = 1;
    
    dX(1) = dx;
    dX(2) = - K_d * (dx - dy_r) - K_p * (x - y_r) + ddy_r; % y'' = e_y'' + ddy_r
    dX(3) = (- K_d * (dx - dy_r) - K_p * (x - y_r)) / V_x; % w 
end