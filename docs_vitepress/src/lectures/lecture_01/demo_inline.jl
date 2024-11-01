fsum(x::Int,p...)=x+fsum(p[1],p[2:end]...)
fsum(x::Int) = x


