
syms xi
xw_ = 0.20;
yw_ = 0.00;
rw_ = 1.20;


x= matlabFunction( (-xw_+rw_*cos(xi)).*(1+1./(rw_^2+xw_^2+yw_^2+2*rw_*(-xw_*cos(xi)+yw_*sin(xi))))/4  );
y= matlabFunction( (+yw_+rw_*sin(xi)).*(1-1./(rw_^2+xw_^2+yw_^2+2*rw_*(-xw_*cos(xi)+yw_*sin(xi))))/4  );



NP = 1000;
xiv = 0:(2*pi/NP):(2*pi);

X = x(xiv);
Y = y(xiv);



deg   = 2;
N_co  = 50;

N_el  = 200;
Th_knts  = [0.125,0.8750];
name = 'WingSection_coarse';

% N_el  = 800;
% Th_knts  = [0.125, 0.375,0.625,0.8750];
% name = 'WingSection';

% N_el  = 2000;
% Th_knts  = 0.1:0.1:0.9;
% name = 'WingSection_dense';




nquad = 3;
Delta = 0.01;
f1 = 0.6;
f2 = 0.6;

xi_al = arc_lenght(X,Y);




curve = l2_proj(xi_al,deg, [X;Y],N_co,nquad);
curve = flat_trailing_edge(curve, f1);
curve.coefs(:,1) = curve.coefs(:,end); 
curve.coefs(1,[1,end]) = [curve.coefs(1,2),curve.coefs(1,end-1)];

new_knts = 0:(1/N_el):1;
new_knts = unique(new_knts(~ismembertol(new_knts,curve.knots,1e-6)));
curve = nrbkntins(curve, new_knts);



off_curve = offset_curve(curve,Delta,deg,N_el,nquad,f2);
off_curve.coefs(:,1) = off_curve.coefs(:,end); 

off_curve.knots = curve.knots;
surf = nrbruled(curve,off_curve); surf = nrbdegelev(surf,[0,1]);
surf = nrbkntins(surf,{[],Th_knts});

nrbkntplot(surf); hold on
view(2)

map = geo_load(curve);
map = map.map;

xiv= 0:(1/1000):1;
Xv = map({xiv});

I = find(Xv(2,:)>0.1085);
disp(min(xiv(I)))
disp(max(xiv(I)))
plot(Xv(1,I),Xv(2,I),'r-','LineWidth',5)


I = find(Xv(2,:)<-0.1085);
disp(min(xiv(I)))
disp(max(xiv(I)))
plot(Xv(1,I),Xv(2,I),'r-','LineWidth',5)


coefs = surf.coefs(1:2,:,:);
knt1 = surf.knots{1};
knt2 = surf.knots{2};


save([name,'.mat'], 'coefs','knt1','knt2');





syms xi xw rw V0 a 

assume(xi, 'real')
assume(xw, 'real')
assume(rw, 'real')
assume(V0, 'real')
assume( a, 'real')

xC = xw+rw*cos(2*pi*xi) + 1i*(rw*sin(2*pi*xi));
mu = xw;
Gam = 4*pi*V0*rw*sin(a);
V = V0*exp(-1i*a) + 1i*Gam/(2*pi*(xC-mu)) - V0*rw^2*exp(1i*a) / (xC-mu)^2;
V = simplify(expand(V));
Vr = real(V);
Vi = imag(V);
Vabs2 = simplify(expand(Vr^2 + Vi^2));
p = simplify(expand((V0^2-Vabs2)))/V0^2;

p = subs(p,a,0.1745);

yw = 0;


xs= (-xw+rw*cos(2*pi*xi)).*(1+1./(rw^2+xw^2+yw^2+2*rw*(-xw*cos(2*pi*xi)+yw*sin(2*pi*xi))))/4;
ys= (+yw+rw*sin(2*pi*xi)).*(1-1./(rw^2+xw^2+yw^2+2*rw*(-xw*cos(2*pi*xi)+yw*sin(2*pi*xi))))/4;

tx = simplify(diff(xs));
ty = simplify(diff(ys));

lam = sqrt(simplify(expand(tx^2+ty^2)));

ny = simplify(expand(subs(-tx /lam , [xw,rw],[xw_,rw_])));
nx = simplify(expand(subs( ty /lam , [xw,rw],[xw_,rw_])));

t = subs([nx;ny], [xw,rw],[xw_,rw_]);

P = matlabFunction(p);




xiv = 0:(1/1000):1;
crv = geo_load(curve);
X  = zeros(2,length(xiv));
Xt = zeros(2,length(xiv));
for i=1:length(xiv)
    X (:,i) = crv.map({xiv(i)});
    
    n = crv.map_der({xiv(i)});
    n = [n(2);-n(1)];
    n = n/norm(n);
    Xt(:,i) = X(:,i) + P(xiv(i))*n;
end

curv_force = l2_proj(xiv,deg,Xt-X,N_el,nquad);

coefs = curv_force.coefs(1:2,:,:);
knt   = curv_force.knots;
%save('WingForce.mat', 'coefs','knt');




figure
% plot( X(1,:), X(2,:)); hold on; axis equal
% plot(Xt(1,:),Xt(2,:))

plot(Xt(1,:)-X(1,:),Xt(2,:)-X(2,:)); hold on; axis equal













% Local functions =========================================================
function xi_al = arc_lenght(X,Y)
    xi_al    = zeros(size(X));
    xi_al(1) = 0;
    for i = 1:(length(X)-1)
        xi_al(i+1) = xi_al(i) + sqrt( (X(i+1)-X(i))^2 + (Y(i+1)-Y(i))^2);
    end
end
function curve = l2_proj(tv,deg,X,N_el,nquad)
    crv = nrbline([0,0,0],[1,0,0]); crv.knots((end-1):end) = tv(end); crv = nrbdegelev(crv,(deg-1));
    crv = nrbkntins(crv,(tv(end)/N_el):(tv(end)/N_el):(tv(end)*(N_el-1)/N_el));
    zeta = unique(crv.knots);
    crv.coefs(2,:) = 1;
    geometry = geo_load(crv);
    rule     = msh_gauss_nodes (nquad);
    [qn, qw] = msh_set_quad_nodes (zeta, rule);
    mesh     = msh_cartesian (zeta, qn, qw, geometry);
    space    = sp_bspline (crv.knots, deg, mesh);
    space    = sp_vector ({space,space}, mesh);
    mesh_ev  = msh_evaluate_element_list(mesh,1:N_el);
    space_ev = sp_evaluate_element_list(space,mesh_ev);
    coeff_ev = coefficients(mesh_ev,tv,X);
    mat = op_u_v (space_ev, space_ev, mesh_ev, 1);
    rhs = op_f_v (space_ev, mesh_ev,  coeff_ev);
    u   = mat\ rhs;
    crv.coefs(1:3,:) = 0;
    geometry  = geo_load(crv);    
    new_geometry = geo_deform (u, space, geometry);
    curve = new_geometry.nurbs;
    curve.knots(:)   = curve.knots(:)/curve.knots(end);
end
function coeff = coefficients(mesh,tv,X)    
    nodes = reshape(mesh.quad_nodes,mesh.nqn_dir,mesh.nel_dir);
    coeff = zeros([size(X,1),size(nodes)]);   
    for i1 = 1:size(nodes,1)
        for i2 = 1:size(nodes,2)
            xi = nodes(i1,i2);
            ida = sum(xi>tv);
            idb = ida+1;
            ta = tv(ida);
            tb = tv(idb);
            eta = (xi-ta)/(tb-ta);
            coeff(:,i1,i2) = X(:,ida) + eta*(X(:,idb)-X(:,ida));
        end
    end
end
function crv = flat_trailing_edge(crv, f)
    f   = f/2;
    ICP = round(crv.number*f);
    CP1 = crv.coefs(1:2,  1);
    CPf = crv.coefs(1:2,ICP);
    for i = 1:ICP
        crv.coefs(1:2,i) = CP1 + (CPf - CP1)*(i-1)/(ICP-1);
    end
    NCP = crv.number;
    CP1 = crv.coefs(1:2, NCP-ICP+1);
    CPf = crv.coefs(1:2, NCP);
    for i = 1:ICP
        ik = NCP -ICP +i;
        crv.coefs(1:2,ik) = CP1 + (CPf - CP1)*(i-1)/(ICP-1);
    end
end
function off_curve = offset_curve(crv,Delta,deg,N_el,nquad,f)
    geo = geo_load(crv);
    xi = 0:(1/1000):1;
    [Xv,Tv] = geo.map_der({xi});
    lam = sqrt(Tv(1,:).^2+Tv(2,:).^2);
    Tv = Tv./reshape(lam, [1, size(lam)]);
    Tv = [Tv(2,:);-Tv(1,:)];
    X  = Xv + Delta*Tv;
    off_curve = l2_proj(xi,deg, X,N_el,nquad);

%     geo = geo_load(off_curve);
%     [Xv,Tv] = geo.map_der({1/1000});
%     lam = sqrt(Tv(1,:).^2+Tv(2,:).^2);
%     Tv = Tv./reshape(lam, [1, size(lam)]);
%     TP = [Xv(1) + abs(Tv(1)*Xv(2)/Tv(2));0];

%     f   = f/2;
%     ICP = round(off_curve.number*f);
%     CP1 = TP;
%     CPf = off_curve.coefs(1:2,ICP);
%     for i = 1:ICP
%         off_curve.coefs(1:2,i) = CP1 + (CPf - CP1)*(i-1)/(ICP-1);
%     end
%     NCP = off_curve.number;
%     CP1 = off_curve.coefs(1:2, NCP-ICP+1);
%     CPf = TP;
%     for i = 1:ICP
%         ik = NCP -ICP +i;
%         off_curve.coefs(1:2,ik) = CP1 + (CPf - CP1)*(i-1)/(ICP-1);
%     end
end

function C = setdiff_tol(A, B, tol)
    A = A(:);
    B = B(:);
    keep = true(size(A));
    for i = 1:numel(A)
        if any(abs(A(i) - B) <= tol)
            keep(i) = false;
        end
    end
    C = A(keep);
end
















%{
syms xi
assume(xi1, 'real')
xC = xw+rw*cos(2*pi*xi1) + 1i*(yw +rw*sin(2*pi*xi1));
mu = xw + 1i*yw;
a   = 0.1745;
a0  = asin(yw/rw);
Gam = 4*pi*V0*rw*sin(a+a0);
V = V0*exp(-1i*a) + 1i*Gam/(2*pi*(xC-mu)) - V0*rw^2*exp(1i*a) / (xC-mu)^2;
V = expand(V);
Vr = real(V);
Vi = imag(V);
%}





