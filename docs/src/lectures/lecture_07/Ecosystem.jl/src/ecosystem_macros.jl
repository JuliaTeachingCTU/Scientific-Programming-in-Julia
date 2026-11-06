### species definition
macro species(typ, name, icon)
    esc(_species(typ, name, icon))
end

function _species(typ, name, icon)
    quote
            abstract type $name <: $(typ == :Animal ? EcosystemCore.AnimalSpecies : EcosystemCore.PlantSpecies) end
        Base.show(io::IO, ::Type{$name}) = print(io, $(QuoteNode(icon)))
        export $name
    end
end

macro plant(name, icon)
    return :(@species Plant $name $icon)
end

macro animal(name, icon)
    return :(@species Animal $name $icon)
end

### eating behavior
macro eats(species::Symbol, foodlist::Expr)
    return esc(_eats(species, foodlist))
end


function _generate_eat(eater::Type{<:EcosystemCore.AnimalSpecies}, food::Type{<:EcosystemCore.PlantSpecies}, multiplier)
    quote
        EcosystemCore.eats(::EcosystemCore.Animal{$(eater)}, ::EcosystemCore.Plant{$(food)}) = true
        function EcosystemCore.eat!(a::EcosystemCore.Animal{$(eater)}, p::EcosystemCore.Plant{$(food)}, w::EcosystemCore.World)
            if size(p)>0
                EcosystemCore.incr_energy!(a, $(multiplier)*size(p)*EcosystemCore.Δenergy(a))
                p.size = 0
            end
        end
    end
end

function _generate_eat(eater::Type{<:EcosystemCore.AnimalSpecies}, food::Type{<:EcosystemCore.AnimalSpecies}, multiplier)
    quote
        EcosystemCore.eats(::EcosystemCore.Animal{$(eater)}, ::EcosystemCore.Animal{$(food)}) = true
        function EcosystemCore.eat!(ae::EcosystemCore.Animal{$(eater)}, af::EcosystemCore.Animal{$(food)}, w::EcosystemCore.World)
            EcosystemCore.incr_energy!(ae, $(multiplier)*EcosystemCore.energy(af)*EcosystemCore.Δenergy(ae))
            EcosystemCore.kill_agent!(af, w)
        end
    end
end

_parse_eats(ex) = Dict(arg.args[2] => arg.args[3] for arg in ex.args if arg.head == :call && arg.args[1] == :(=>))

function _eats(species, foodlist)
    cfg = _parse_eats(foodlist)
    code = Expr(:block)
    for (k,v) in cfg
        push!(code.args, _generate_eat(eval(species), eval(k), v))
    end
    code
end
