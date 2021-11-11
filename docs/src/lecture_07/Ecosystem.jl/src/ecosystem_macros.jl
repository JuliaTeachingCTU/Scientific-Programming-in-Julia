### species definition
macro plant(name, icon)
    return esc(_species(name, icon; animal=false))
end

macro animal(name, icon)
    return esc(_species(name, icon; animal=true))
end

function _species(name, icon; animal=false)
    quote
        abstract type $name <: $(animal ? AnimalSpecies : PlantSpecies) end
        Base.show(io::IO, ::Type{$name}) = print(io, $(QuoteNode(icon)))
        export $name
    end
end


### eating
macro eats(species::Symbol, foodlist::Expr)
    return esc(_eats(species, foodlist))
end

_parse_eats(ex) = Dict(arg.args[2] => arg.args[3] for arg in ex.args if arg.head == :call && arg.args[1] == :(=>))
function _generate_eat(eater::Type{<:AnimalSpecies}, food::Type{<:PlantSpecies}, multiplier)
    quote
        EcosystemCore.eats(::Animal{$(eater)}, ::Plant{$(food)}) = true
        function EcosystemCore.eat!(a::Animal{$(eater)}, p::Plant{$(food)}, w::World)
            if size(p)>0
                incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))
                p.size = 0
            end
        end
    end
end

function _generate_eat(eater::Type{<:AnimalSpecies}, food::Type{<:AnimalSpecies}, multiplier)
    quote
        EcosystemCore.eats(::Animal{$(eater)}, ::Animal{$(food)}) = true
        function EcosystemCore.eat!(ae::Animal{$(eater)}, af::Animal{$(food)}, w::World)
            incr_energy!(ae, $(multiplier)*energy(af)*Δenergy(ae))
            kill_agent!(af, w)
        end
    end
end

function _eats(species, foodlist)
    cfg = _parse_eats(foodlist)
    code = Expr(:block)
    for (k,v) in cfg
        push!(code.args, _generate_eat(eval(species), eval(k), v))
    end
    code
end