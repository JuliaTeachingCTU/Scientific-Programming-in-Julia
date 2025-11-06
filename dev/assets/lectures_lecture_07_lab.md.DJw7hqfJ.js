import{_ as a,c as n,o as e,ai as p}from"./chunks/framework.BHQ3CDIQ.js";const k=JSON.parse('{"title":"Lab 07: Macros","description":"","frontmatter":{},"headers":[],"relativePath":"lectures/lecture_07/lab.md","filePath":"lectures/lecture_07/lab.md","lastUpdated":null}'),i={name:"lectures/lecture_07/lab.md"};function l(t,s,o,c,h,r){return e(),n("div",null,[...s[0]||(s[0]=[p(`<h1 id="macro_lab" tabindex="-1">Lab 07: Macros <a class="header-anchor" href="#macro_lab" aria-label="Permalink to &quot;Lab 07: Macros {#macro_lab}&quot;">​</a></h1><p>A little reminder from the <a href="/Scientific-Programming-in-Julia/dev/lectures/lecture_07/lecture#macro_lecture">lecture</a>, a macro in its essence is a function, which</p><ol><li><p>takes as an input an expression (parsed input)</p></li><li><p>modifies the expressions in arguments</p></li><li><p>inserts the modified expression at the same place as the one that is parsed.</p></li></ol><p>In this lab we are going to use what we have learned about manipulation of expressions and explore avenues of where macros can be useful</p><ul><li><p>convenience (<code>@repeat</code>, <code>@show</code>)</p></li><li><p>performance critical code generation (<code>@poly</code>)</p></li><li><p>alleviate tedious code generation (<code>@species</code>, <code>@eats</code>)</p></li><li><p>just as a syntactic sugar (<code>@ecosystem</code>)</p></li></ul><h2 id="Show-macro" tabindex="-1">Show macro <a class="header-anchor" href="#Show-macro" aria-label="Permalink to &quot;Show macro {#Show-macro}&quot;">​</a></h2><p>Let&#39;s start with dissecting &quot;simple&quot; <code>@show</code> macro, which allows us to demonstrate advanced concepts of macros and expression manipulation.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @show</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">       # creates a temporary local variable</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">           println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x + 1 = &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">           y               </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># show macro also returns the result</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">       end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"> # assignments should create the variable</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">       @show</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 3</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 3</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> let</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">           println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x = 2 = &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">           y</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">       end</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x                   </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># should be equal to 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span></code></pre></div><p>The original Julia&#39;s <a href="https://github.com/JuliaLang/julia/blob/ae8452a9e0b973991c30f27beb2201db1b0ea0d3/base/show.jl#L946-L959" target="_blank" rel="noreferrer">implementation</a> is not dissimilar to the following macro definition:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">macro</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> myshow</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    quote</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">$</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">QuoteNode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex)), </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot; = &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repr</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> local</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> value </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> $</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">esc</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ex)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        value</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;">@myshow (macro with 1 method)</span></span></code></pre></div><p>Testing it gives us the expected behavior</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @myshow</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">xx </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xx                  </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># should be defined</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span></code></pre></div><p>In this &quot;simple&quot; example, we had to use the following concepts mentioned already in the <a href="/Scientific-Programming-in-Julia/dev/lectures/lecture_07/lecture#macro_lecture">lecture</a>:</p><ul><li><p><code>QuoteNode(ex)</code> is used to wrap the expression inside another layer of quoting, such that when it is interpolated into <code>:()</code> it stays being a piece of code instead of the value it represents - <a href="./@ref lec7_quotation"><strong>TRUE QUOTING</strong></a></p></li><li><p><code>esc(ex)</code> is used in case that the expression contains an assignment, that has to be evaluated in the top level module <code>Main</code> (we are <code>esc</code>aping the local context) - <a href="./@ref lec7_hygiene"><strong>ESCAPING</strong></a></p></li><li><p><code>$(QuoteNode(ex))</code> and <code>$(esc(ex))</code> is used to evaluate an expression into another expression. <a href="./@ref lec7_quotation"><strong>INTERPOLATION</strong></a></p></li><li><p><code>local value =</code> is used in order to return back the result after evaluation</p></li></ul><p>Lastly, let&#39;s mention that we can use <code>@macroexpand</code> to see how the code is manipulated in the <code>@myshow</code> macro</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">julia</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @macroexpand</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @show</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">quote</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;x + 1 = &quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, Base</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repr</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                \x1B[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">90</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">#= show.jl:1270 =#</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">\x1B[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">39</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">m</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                local</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> var&quot;#228#value&quot;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    var&quot;#228#value&quot;</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span></span></code></pre></div><h2 id="Repeat-macro" tabindex="-1">Repeat macro <a class="header-anchor" href="#Repeat-macro" aria-label="Permalink to &quot;Repeat macro {#Repeat-macro}&quot;">​</a></h2><p>In the profiling/performance <a href="/Scientific-Programming-in-Julia/dev/lectures/lecture_05/lab#perf_lab">labs</a> we have sometimes needed to run some code multiple times in order to gather some samples and we have tediously written out simple for loops inside functions such as this</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> run_polynomial</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(n, a, x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">n</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        polynomial</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(a, x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We can remove this boilerplate code by creating a very simple macro that does this for us. ::: warning &quot;Exercise&quot;</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>    Define macro \`@repeat\` that takes two arguments, first one being the number of times a code is to be run and the other being the actual code.</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    julia&gt; @repeat 3 println(&quot;Hello!&quot;)</span></span>
<span class="line"><span>    Hello!</span></span>
<span class="line"><span>    Hello!</span></span>
<span class="line"><span>    Hello!</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    Before defining the macro, it is recommended to write the code manipulation functionality into a helper function \`_repeat\`, which helps in organization and debugging of macros.</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    _repeat(3, :(println(&quot;Hello!&quot;))) # testing &quot;macro&quot; without defining it</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **HINTS**:</span></span>
<span class="line"><span>    - use \`$\` interpolation into a for loop expression; for example given \`ex = :(1+x)\` we can interpolate it into another expression \`:($ex + y)\` -&gt; \`:(1 + x + y)\`</span></span>
<span class="line"><span>    - if unsure what gets interpolated use round brackets \`:($(ex) + y)\`</span></span>
<span class="line"><span>    - macro is a function that *creates* code that does what we want</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **BONUS**:</span></span>
<span class="line"><span>    What happens if we call \`@repeat 3 x = 2\`? Is \`x\` defined?</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: details</span></span>
<span class="line"><span>    \`\`\`@repl lab07_repeat</span></span>
<span class="line"><span>    macro repeat(n::Int, ex)</span></span>
<span class="line"><span>        return _repeat(n, ex)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _repeat(n::Int, ex)</span></span>
<span class="line"><span>        :(for _ in 1:$n</span></span>
<span class="line"><span>            $ex</span></span>
<span class="line"><span>         end)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    _repeat(3, :(println(&quot;Hello!&quot;)))</span></span>
<span class="line"><span>    @repeat 3 println(&quot;Hello!&quot;)</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    Even if we had used escaping the expression \`x = 2\` won&#39;t get evaluated properly due to the induced scope of the for loop. In order to resolve this we would have to specially match that kind of expression and generate a proper syntax withing the for loop \`global $ex\`. However we may just warn the user in the docstring that the usage is disallowed. </span></span>
<span class="line"><span></span></span>
<span class="line"><span>Note that this kind of repeat macro is also defined in the [\`Flux.jl\`](https://fluxml.ai/) machine learning framework, wherein it&#39;s called \`@epochs\` and is used for creating training [loop](https://fluxml.ai/Flux.jl/stable/training/training/#Datasets).</span></span>
<span class="line"><span></span></span>
<span class="line"><span>## [Polynomial macro](@id lab07_polymacro)</span></span>
<span class="line"><span>This is probably the last time we are rewriting the \`polynomial\` function, though not quite in the same way. We have seen in the last [lab](@ref introspection_lab), that some optimizations occur automatically, when the compiler can infer the length of the coefficient array, however with macros we can *generate* optimized code directly (not on the same level - we are essentially preparing already unrolled/inlined code).</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Ideally we would like to write some macro \`@poly\` that takes a polynomial in a mathematical notation and spits out an anonymous function for its evaluation, where the loop is unrolled. </span></span>
<span class="line"><span></span></span>
<span class="line"><span>*Example usage*:</span></span></code></pre></div><p>julia p = @poly x 3x^2+2x^1+10x^0 # the first argument being the independent variable to match p(2) # return the value</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span></span></span>
<span class="line"><span>However in order to make this happen, let&#39;s first consider much simpler case of creating the same but without the need for parsing the polynomial as a whole and employ the fact that macro can have multiple arguments separated by spaces.</span></span></code></pre></div><p>julia p = @poly 3 2 10 p(2)</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span></span></span>
<span class="line"><span>::: warning &quot;Exercise&quot;</span></span>
<span class="line"><span>    Create macro \`@poly\` that takes multiple arguments and creates an anonymous function that constructs the unrolled code. Instead of directly defining the macro inside the macro body, create helper function \`_poly\` with the same signature that can be reused outside of it.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    Recall Horner&#39;s method polynomial evaluation from previous [labs](@ref horner):</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    function polynomial(a, x)</span></span>
<span class="line"><span>        accumulator = a[end] * one(x)</span></span>
<span class="line"><span>        for i in length(a)-1:-1:1</span></span>
<span class="line"><span>            accumulator = accumulator * x + a[i]</span></span>
<span class="line"><span>            #= accumulator = muladd(x, accumulator, a[i]) =# # equivalent</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        accumulator  </span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **HINTS**:</span></span>
<span class="line"><span>    - you can use \`muladd\` function as replacement for \`ac * x + a[i]\`</span></span>
<span class="line"><span>    - think of the \`accumulator\` variable as the mathematical expression that is incrementally built (try to write out the Horner&#39;s method[^1] to see it)</span></span>
<span class="line"><span>    - you can nest expression arbitrarily</span></span>
<span class="line"><span>    - the order of coefficients has different order than in previous labs (going from high powers of \`x\` last to them being first)</span></span>
<span class="line"><span>    - use \`evalpoly\` to check the correctness</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    using Test</span></span>
<span class="line"><span>    p = @poly 3 2 10</span></span>
<span class="line"><span>    @test p(2) == evalpoly(2, [10,2,3]) # reversed coefficients</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>[^1]: Explanation of the Horner schema can be found on [https://en.wikipedia.org/wiki/Horner%27s\\_method](https://en.wikipedia.org/wiki/Horner%27s_method).</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: details</span></span>
<span class="line"><span>    \`\`\`@repl lab07_poly</span></span>
<span class="line"><span>    using InteractiveUtils #hide</span></span>
<span class="line"><span>    macro poly(a...)</span></span>
<span class="line"><span>        return _poly(a...)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _poly(a...)</span></span>
<span class="line"><span>        N = length(a)</span></span>
<span class="line"><span>        ex = :($(a[1]))</span></span>
<span class="line"><span>        for i in 2:N</span></span>
<span class="line"><span>            ex = :(muladd(x, $ex, $(a[i]))) # equivalent of :(x * $ex + $(a[i]))</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        :(x -&gt; $ex)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    p = @poly 3 2 10</span></span>
<span class="line"><span>    p(2) == evalpoly(2, [10,2,3])</span></span>
<span class="line"><span>    @code_lowered p(2) # can show the generated code</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Moving on to the first/harder case, where we need to parse the mathematical expression.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: warning &quot;Exercise&quot;</span></span>
<span class="line"><span>    Create macro \`@poly\` that takes two arguments first one being the independent variable and second one being the polynomial written in mathematical notation. As in the previous case this macro should define an anonymous function that constructs the unrolled code. </span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    julia&gt; p = @poly x 3x^2+2x^1+10x^0  # the first argument being the independent variable to match</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **HINTS**:</span></span>
<span class="line"><span>    - though in general we should be prepared for some edge cases, assume that we are really strict with the syntax allowed (e.g. we really require spelling out x^0, even though it is mathematically equivalent to \`1\`)</span></span>
<span class="line"><span>    - reuse the \`_poly\` function from the previous exercise</span></span>
<span class="line"><span>    - use the \`MacroTools.jl\` to match/capture \`a_*$v^(n_)\`, where \`v\` is the symbol of independent variable, this is going to be useful in the following steps</span></span>
<span class="line"><span>        1. get maximal rank of the polynomial</span></span>
<span class="line"><span>        2. get coefficient for each power</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    ::: note &quot;\`MacroTools.jl\`&quot;</span></span>
<span class="line"><span>        Though not the most intuitive, [\`MacroTools.jl\`](https://fluxml.ai/MacroTools.jl/stable/) pkg help us with writing custom macros. We will use two utilities</span></span>
<span class="line"><span>        #### \`@capture\`</span></span>
<span class="line"><span>        This macro is used to match a pattern in a *single* expression and return values of particular spots. For example</span></span>
<span class="line"><span>        \`\`\`julia</span></span>
<span class="line"><span>        julia&gt; using MacroTools</span></span>
<span class="line"><span>        julia&gt; @capture(:[1, 2, 3, 4, 5, 6, 7], [1, a_, 3, b__, c_])</span></span>
<span class="line"><span>        true</span></span>
<span class="line"><span>        </span></span>
<span class="line"><span>        julia&gt; a, b, c</span></span>
<span class="line"><span>        (2,[4,5,6],7)</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span>        #### \`postwalk\`/\`prewalk\`</span></span>
<span class="line"><span>        In order to extend \`@capture\` to more complicated expression trees, we can used either \`postwalk\` or \`prewalk\` to walk the AST and match expression along the way. For example</span></span>
<span class="line"><span>        \`\`\`julia</span></span>
<span class="line"><span>        julia&gt; using MacroTools: prewalk, postwalk</span></span>
<span class="line"><span>        julia&gt; ex = quote</span></span>
<span class="line"><span>            x = f(y, g(z))</span></span>
<span class="line"><span>            return h(x)</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        </span></span>
<span class="line"><span>        julia&gt; postwalk(ex) do x</span></span>
<span class="line"><span>                @capture(x, fun_(arg_)) &amp;&amp; println(&quot;Function: &quot;, fun, &quot; with argument: &quot;, arg)</span></span>
<span class="line"><span>                x</span></span>
<span class="line"><span>            end;</span></span>
<span class="line"><span>        Function: g with argument: z</span></span>
<span class="line"><span>        Function: h with argument: x</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span>        Note that the \`x\` or the iteration is required, because by default postwalk/prewalk replaces currently read expression with the output of the body of \`do\` block.</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: details</span></span>
<span class="line"><span>    \`\`\`@example lab07_poly</span></span>
<span class="line"><span>    using MacroTools</span></span>
<span class="line"><span>    using MacroTools: postwalk, prewalk</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    macro poly(v::Symbol, p::Expr)</span></span>
<span class="line"><span>        a = Tuple(reverse(_get_coeffs(v, p)))</span></span>
<span class="line"><span>        return _poly(a...)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _max_rank(v, p)</span></span>
<span class="line"><span>        mr = 0</span></span>
<span class="line"><span>        postwalk(p) do x</span></span>
<span class="line"><span>            if @capture(x, a_*$v^(n_))</span></span>
<span class="line"><span>                mr = max(mr, n)</span></span>
<span class="line"><span>            end</span></span>
<span class="line"><span>            x</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        mr</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _get_coeffs(v, p)</span></span>
<span class="line"><span>        N = _max_rank(v, p) + 1</span></span>
<span class="line"><span>        coefficients = zeros(N)</span></span>
<span class="line"><span>        postwalk(p) do x</span></span>
<span class="line"><span>            if @capture(x, a_*$v^(n_))</span></span>
<span class="line"><span>                coefficients[n+1] = a</span></span>
<span class="line"><span>            end</span></span>
<span class="line"><span>            x</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        coefficients</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    Let&#39;s test it.</span></span>
<span class="line"><span>    \`\`\`@repl lab07_poly</span></span>
<span class="line"><span>    p = @poly x 3x^2+2x^1+10x^0</span></span>
<span class="line"><span>    p(2) == evalpoly(2, [10,2,3])</span></span>
<span class="line"><span>    @code_lowered p(2) # can show the generated code</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>## Ecosystem macros</span></span>
<span class="line"><span>There are at least two ways how we can make our life simpler when using our \`Ecosystem\` and \`EcosystemCore\` pkgs. Firstly, recall that in order to test our simulation we always had to write something like this:</span></span></code></pre></div><p>julia function create_world() n_grass = 500 regrowth_time = 17.0</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>n_sheep         = 100</span></span>
<span class="line"><span>Δenergy_sheep   = 5.0</span></span>
<span class="line"><span>sheep_reproduce = 0.5</span></span>
<span class="line"><span>sheep_foodprob  = 0.4</span></span>
<span class="line"><span></span></span>
<span class="line"><span>n_wolves       = 8</span></span>
<span class="line"><span>Δenergy_wolf   = 17.0</span></span>
<span class="line"><span>wolf_reproduce = 0.03</span></span>
<span class="line"><span>wolf_foodprob  = 0.02</span></span>
<span class="line"><span></span></span>
<span class="line"><span>gs = [Grass(id, regrowth_time) for id in 1:n_grass];</span></span>
<span class="line"><span>ss = [Sheep(id, 2*Δenergy_sheep, Δenergy_sheep, sheep_reproduce, sheep_foodprob) for id in n_grass+1:n_grass+n_sheep];</span></span>
<span class="line"><span>ws = [Wolf(id, 2*Δenergy_wolf, Δenergy_wolf, wolf_reproduce, wolf_foodprob) for id in n_grass+n_sheep+1:n_grass+n_sheep+n_wolves];</span></span>
<span class="line"><span>World(vcat(gs, ss, ws))</span></span></code></pre></div><p>end world = create_world();</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>which includes the tedious process of defining the agent counts, their parameters and last but not least the unique id manipulation. As part of the [HW](@ref hw07) for this lecture you will be tasked to define a simple DSL, which can be used to define a world in a few lines.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Secondly, the definition of a new \`Animal\` or \`Plant\`, that did not have any special behavior currently requires quite a bit of repetitive code. For example defining a new plant type \`Broccoli\` goes as follows</span></span></code></pre></div><p>julia abstract type Broccoli &lt;: PlantSpecies end Base.show(io::IO,::Type{Broccoli}) = print(io,&quot;🥦&quot;)</p><p>EcosystemCore.eats(::Animal{Sheep},::Plant{Broccoli}) = true</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span></span></span>
<span class="line"><span>and definition of a new animal like a \`Rabbit\` looks very similar</span></span></code></pre></div><p>julia abstract type Rabbit &lt;: AnimalSpecies end Base.show(io::IO,::Type{Rabbit}) = print(io,&quot;🐇&quot;)</p><p>EcosystemCore.eats(::Animal{Rabbit},p::Plant{Grass}) = size(p) &gt; 0 EcosystemCore.eats(::Animal{Rabbit},p::Plant{Broccoli}) = size(p) &gt; 0</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>In order to make this code &quot;clearer&quot; (depends on your preference) we will create two macros, which can be called at one place to construct all the relations.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>### New Animal/Plant definition</span></span>
<span class="line"><span>Our goal is to be able to define new plants and animal species, while having a clear idea about their relations. For this we have proposed the following macros/syntax:</span></span></code></pre></div><p>julia @species Plant Broccoli 🥦 @species Animal Rabbit 🐇 @eats Rabbit [Grass =&gt; 0.5, Broccoli =&gt; 1.0, Mushroom =&gt; -1.0]</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Unfortunately the current version of \`Ecosystem\` and \`EcosystemCore\`, already contains some definitions of species such as \`Sheep\`, \`Wolf\` and \`Mushroom\`, which may collide with definitions during prototyping, therefore we have created a modified version of those pkgs, which will be provided in the lab.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: note &quot;Testing relations&quot;</span></span>
<span class="line"><span>    We can test the current definition with the following code that constructs &quot;eating matrix&quot;</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    using Ecosystem</span></span>
<span class="line"><span>    using Ecosystem.EcosystemCore</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    function eating_matrix()</span></span>
<span class="line"><span>        _init(ps::Type{&lt;:PlantSpecies}) = ps(1, 10.0)</span></span>
<span class="line"><span>        _init(as::Type{&lt;:AnimalSpecies}) = as(1, 10.0, 1.0, 0.8, 0.7)</span></span>
<span class="line"><span>        function _check(s1, s2)</span></span>
<span class="line"><span>            try</span></span>
<span class="line"><span>                if s1 !== s2</span></span>
<span class="line"><span>                    EcosystemCore.eats(_init(s1), _init(s2)) ? &quot;✅&quot; : &quot;❌&quot;</span></span>
<span class="line"><span>                else</span></span>
<span class="line"><span>                    return &quot;❌&quot;</span></span>
<span class="line"><span>                end</span></span>
<span class="line"><span>            catch e</span></span>
<span class="line"><span>                if e isa MethodError</span></span>
<span class="line"><span>                    return &quot;❔&quot;</span></span>
<span class="line"><span>                else</span></span>
<span class="line"><span>                    throw(e)</span></span>
<span class="line"><span>                end</span></span>
<span class="line"><span>            end</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>        animal_species = subtypes(AnimalSpecies)</span></span>
<span class="line"><span>        plant_species = subtypes(PlantSpecies)</span></span>
<span class="line"><span>        species = vcat(animal_species, plant_species)</span></span>
<span class="line"><span>        em = [_check(s, ss) for (s,ss) in Iterators.product(animal_species, species)]</span></span>
<span class="line"><span>        string.(hcat([&quot;🌍&quot;, animal_species...], vcat(permutedims(species), em)))</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span>    eating_matrix()</span></span>
<span class="line"><span>     🌍  🐑  🐺  🌿  🍄</span></span>
<span class="line"><span>     🐑  ❌  ❌  ✅  ✅</span></span>
<span class="line"><span>     🐺  ✅  ❌  ❌  ❌</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>::: warning &quot;Exercise&quot;</span></span>
<span class="line"><span>    Based on the following example syntax, </span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    @species Plant Broccoli 🥦</span></span>
<span class="line"><span>    @species Animal Rabbit 🐇</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    write macro \`@species\` inside \`Ecosystem\` pkg, which defines the abstract type, its show function and exports the type. For example \`@species Plant Broccoli 🥦\` should generate code:</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    abstract type Broccoli &lt;: PlantSpecies end</span></span>
<span class="line"><span>    Base.show(io::IO,::Type{Broccoli}) = print(io,&quot;🥦&quot;)</span></span>
<span class="line"><span>    export Broccoli</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    Define first helper function \`_species\` to inspect the macro&#39;s output. This is indispensable, as we are defining new types/constants and thus we may otherwise encounter errors during repeated evaluation (though only if the type signature changed).</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    _species(:Plant, :Broccoli, :🥦)</span></span>
<span class="line"><span>    _species(:Animal, :Rabbit, :🐇)</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: warning &quot;Exercise&quot;</span></span>
<span class="line"><span>    Based on the following example syntax, </span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    @species Plant Broccoli 🥦</span></span>
<span class="line"><span>    @species Animal Rabbit 🐇</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    write macro \`@species\` inside \`Ecosystem\` pkg, which defines the abstract type, its show function and exports the type. For example \`@species Plant Broccoli 🥦\` should generate code:</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    abstract type Broccoli &lt;: PlantSpecies end</span></span>
<span class="line"><span>    Base.show(io::IO,::Type{Broccoli}) = print(io,&quot;🥦&quot;)</span></span>
<span class="line"><span>    export Broccoli</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    Define first helper function \`_species\` to inspect the macro&#39;s output. This is indispensable, as we are defining new types/constants and thus we may otherwise encounter errors during repeated evaluation (though only if the type signature changed).</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    _species(:Plant, :Broccoli, :🥦)</span></span>
<span class="line"><span>    _species(:Animal, :Rabbit, :🐇)</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **HINTS**:</span></span>
<span class="line"><span>    - use \`QuoteNode\` in the show function just like in the \`@myshow\` example</span></span>
<span class="line"><span>    - escaping \`esc\` is needed for the returned in order to evaluate in the top most module (\`Ecosystem\`/\`Main\`)</span></span>
<span class="line"><span>    - ideally these changes should be made inside the modified \`Ecosystem\` pkg provided in the lab (though not everything can be refreshed with \`Revise\`) - there is a file \`ecosystem_macros.jl\` just for this purpose</span></span>
<span class="line"><span>    - multiple function definitions can be included into a \`quote end\` block</span></span>
<span class="line"><span>    - interpolation works with any expression, e.g. \`$(typ == :Animal ? AnimalSpecies : PlantSpecies)\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **BONUS**:</span></span>
<span class="line"><span>    Based on \`@species\` define also macros \`@animal\` and \`@plant\` with two arguments instead of three, where the species type is implicitly carried in the macro&#39;s name.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: details</span></span>
<span class="line"><span>    Macro \`@species\`</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    macro species(typ, name, icon)</span></span>
<span class="line"><span>        esc(_species(typ, name, icon))</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _species(typ, name, icon)</span></span>
<span class="line"><span>        quote</span></span>
<span class="line"><span>            abstract type $name &lt;: $(typ == :Animal ? AnimalSpecies : PlantSpecies) end</span></span>
<span class="line"><span>            Base.show(io::IO, ::Type{$name}) = print(io, $(QuoteNode(icon)))</span></span>
<span class="line"><span>            export $name</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    _species(:Plant, :Broccoli, :🥦)</span></span>
<span class="line"><span>    _species(:Animal, :Rabbit, :🐇)</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    And the bonus macros \`@plant\` and \`@animal\`</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    macro plant(name, icon)</span></span>
<span class="line"><span>        return :(@species Plant $name $icon)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    macro animal(name, icon)</span></span>
<span class="line"><span>        return :(@species Animal $name $icon)</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>The next exercise applies macros to the agents eating behavior.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: warning &quot;Exercise&quot;</span></span>
<span class="line"><span>    Define macro \`@eats\` inside \`Ecosystem\` pkg that assigns particular species their eating habits via \`eat!\` and \`eats\` functions. The macro should process the following example syntax</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    @eats Rabbit [Grass =&gt; 0.5, Broccoli =&gt; 1.0],</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span>    where \`Grass =&gt; 0.5\` defines the behavior of the \`eat!\` function. The coefficient is used here as a multiplier for the energy balance, in other words the \`Rabbit\` should get only \`0.5\` of energy for a piece of \`Grass\`.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    **HINTS**:</span></span>
<span class="line"><span>    - ideally these changes should be made inside the modified \`Ecosystem\` pkg provided in the lab (though not everything can be refreshed with \`Revise\`) - there is a file \`ecosystem_macros.jl\` just for this purpose</span></span>
<span class="line"><span>    - escaping \`esc\` is needed for the returned in order to evaluate in the top most module (\`Ecosystem\`/\`Main\`)</span></span>
<span class="line"><span>    - you can create an empty \`quote end\` block with \`code = Expr(:block)\` and push new expressions into its \`args\` incrementally</span></span>
<span class="line"><span>    - use dispatch to create specific code for the different combinations of agents eating other agents (there may be catch in that we have to first \`eval\` the symbols before calling in order to know if they are animals or plants)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    ::: note &quot;Reminder of \`EcosystemCore\` \`eat!\` and \`eats\` functionality&quot;</span></span>
<span class="line"><span>        In order to define that an \`Wolf\` eats \`Sheep\`, we have to define two methods</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span>        EcosystemCore.eats(::Animal{Wolf}, ::Animal{Sheep}) = true</span></span>
<span class="line"><span>        </span></span>
<span class="line"><span>        function EcosystemCore.eat!(ae::Animal{Wolf}, af::Animal{Sheep}, w::World)</span></span>
<span class="line"><span>            incr_energy!(ae, $(multiplier)*energy(af)*Δenergy(ae))</span></span>
<span class="line"><span>            kill_agent!(af, w)</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span>        In order to define that an \`Sheep\` eats \`Grass\`, we have to define two methods</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span>        EcosystemCore.eats(::Animal{Sheep}, p::Plant{Grass}) = size(p)&gt;0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        function EcosystemCore.eat!(a::Animal{Sheep}, p::Plant{Grass}, w::World)</span></span>
<span class="line"><span>            incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))</span></span>
<span class="line"><span>            p.size = 0</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>**BONUS**:</span></span>
<span class="line"><span>You can try running the simulation with the newly added agents.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>::: details</span></span>
<span class="line"><span>    \`\`\`julia</span></span>
<span class="line"><span>    macro eats(species::Symbol, foodlist::Expr)</span></span>
<span class="line"><span>        return esc(_eats(species, foodlist))</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _generate_eat(eater::Type{&lt;:AnimalSpecies}, food::Type{&lt;:PlantSpecies}, multiplier)</span></span>
<span class="line"><span>        quote</span></span>
<span class="line"><span>            EcosystemCore.eats(::Animal{$(eater)}, p::Plant{$(food)}) = size(p)&gt;0</span></span>
<span class="line"><span>            function EcosystemCore.eat!(a::Animal{$(eater)}, p::Plant{$(food)}, w::World)</span></span>
<span class="line"><span>                incr_energy!(a, $(multiplier)*size(p)*Δenergy(a))</span></span>
<span class="line"><span>                p.size = 0</span></span>
<span class="line"><span>            end</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _generate_eat(eater::Type{&lt;:AnimalSpecies}, food::Type{&lt;:AnimalSpecies}, multiplier)</span></span>
<span class="line"><span>        quote</span></span>
<span class="line"><span>            EcosystemCore.eats(::Animal{$(eater)}, ::Animal{$(food)}) = true</span></span>
<span class="line"><span>            function EcosystemCore.eat!(ae::Animal{$(eater)}, af::Animal{$(food)}, w::World)</span></span>
<span class="line"><span>                incr_energy!(ae, $(multiplier)*energy(af)*Δenergy(ae))</span></span>
<span class="line"><span>                kill_agent!(af, w)</span></span>
<span class="line"><span>            end</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    _parse_eats(ex) = Dict(arg.args[2] =&gt; arg.args[3] for arg in ex.args if arg.head == :call &amp;&amp; arg.args[1] == :(=&gt;))</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    function _eats(species, foodlist)</span></span>
<span class="line"><span>        cfg = _parse_eats(foodlist)</span></span>
<span class="line"><span>        code = Expr(:block)</span></span>
<span class="line"><span>        for (k,v) in cfg</span></span>
<span class="line"><span>            push!(code.args, _generate_eat(eval(species), eval(k), v))</span></span>
<span class="line"><span>        end</span></span>
<span class="line"><span>        code</span></span>
<span class="line"><span>    end</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    species = :Rabbit </span></span>
<span class="line"><span>    foodlist = :([Grass =&gt; 0.5, Broccoli =&gt; 1.0])</span></span>
<span class="line"><span>    _eats(species, foodlist)</span></span>
<span class="line"><span>    \`\`\`</span></span>
<span class="line"><span></span></span>
<span class="line"><span>---</span></span>
<span class="line"><span>## Resources</span></span>
<span class="line"><span>- macros in Julia [documentation](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>### \`Type{T}\` type selectors</span></span>
<span class="line"><span>We have used \`::Type{T}\` signature[^2] at few places in the \`Ecosystem\` family of packages (and it will be helpful in the HW as well), such as in the \`show\` methods</span></span></code></pre></div><p>julia Base.show(io::IO,::Type{World}) = print(io,&quot;🌍&quot;) \`\`\`This particular example defines a method where the second argument is the<code>World</code>type itself and not an instance of a<code>World</code> type. As a result we are able to dispatch on specific types as values.</p><p>Furthermore we can use subtyping operator to match all types in a hierarchy, e.g. <code>::Type{&lt;:AnimalSpecies}</code> matches all animal species</p>`,40)])])}const u=a(i,[["render",l]]);export{k as __pageData,u as default};
