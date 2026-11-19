# API reference

Every exported name, by topic. Vector fields are tuples `(ux, uy)` of arrays on the grid; functions are arrays of
grid values or numbers (a constant).

## Discretizations

```@docs
disk
Disk
ellipse
domain
Domain
```

## Basis and transforms

```@docs
ylm
∂ylm∂ζ
Nlm
λlm
Clmn
lm2index
index2lm
psh
psh!
ipsh
ipsh!
psh_matrix
ipsh_matrix
upsample
legpts
trigpts
```

## Integrals, traces and coefficient-space operators

```@docs
integral
trace
apply
solve
```

## Laplace operators

```@docs
𝒮
𝒮⁻¹
𝒩
𝒩⁻¹
𝒮𝒩⁻¹
Δ⁻¹
```

## Volume potentials

```@docs
𝒱
ℬ
𝒯
```

## Shape derivatives

```@docs
δ𝒮
δ𝒩
δ𝒱
δℬ
```

## Differential operators

```@docs
∂ζ
∂ζ̄
∂x
∂y
grad
div(::Tuple, ::Disk)
lap
∂n
```

## Stokes flow

```@docs
𝒮_st
𝒮_st⁻¹
stokes_mobility_matrix
stokes_mobility_solve
```

## Suspensions

```@docs
suspension_mobility_solve
suspension_velocity_solve
suspension_velocity_eval
suspension_self_matrix
```

## Utilities

```@docs
gmres
aca
```
