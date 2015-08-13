  .text
  .globl realloc
  .type realloc, @function

#! file-offset 0x1571c0
#! rip-offset  0x1171c0
#! capacity    704 bytes

# Text                         #  Line  RIP       Bytes  Opcode              
.realloc:                      #        0x1171c0  0      OPC=<label>         
  movq %r12, -0x18(%rsp)       #  1     0x1171c0  5      OPC=movq_m64_r64    
  movl %edi, %r12d             #  2     0x1171c5  3      OPC=movl_r32_r32    
  movq %rbx, -0x20(%rsp)       #  3     0x1171c8  5      OPC=movq_m64_r64    
  movq %r13, -0x10(%rsp)       #  4     0x1171cd  5      OPC=movq_m64_r64    
  movq %r14, -0x8(%rsp)        #  5     0x1171d2  5      OPC=movq_m64_r64    
  subl $0x38, %esp             #  6     0x1171d7  3      OPC=subl_r32_imm8   
  addq %r15, %rsp              #  7     0x1171da  3      OPC=addq_r64_r64    
  testq %r12, %r12             #  8     0x1171dd  3      OPC=testq_r64_r64   
  movl %esi, %ebx              #  9     0x1171e0  2      OPC=movl_r32_r32    
  je .L_117400                 #  10    0x1171e2  6      OPC=je_label_1      
  cmpl $0xffffffbf, %esi       #  11    0x1171e8  6      OPC=cmpl_r32_imm32  
  nop                          #  12    0x1171ee  1      OPC=nop             
  nop                          #  13    0x1171ef  1      OPC=nop             
  nop                          #  14    0x1171f0  1      OPC=nop             
  ja .L_117440                 #  15    0x1171f1  6      OPC=ja_label_1      
  cmpl $0xa, %ebx              #  16    0x1171f7  3      OPC=cmpl_r32_imm8   
  movl $0x10, %esi             #  17    0x1171fa  5      OPC=movl_r32_imm32  
  ja .L_117320                 #  18    0x1171ff  6      OPC=ja_label_1      
  nop                          #  19    0x117205  1      OPC=nop             
  testb $0x2, 0xff61b35(%rip)  #  20    0x117206  7      OPC=testb_m8_imm8   
  jne .L_1172c0                #  21    0x11720d  6      OPC=jne_label_1     
  nop                          #  22    0x117213  1      OPC=nop             
  nop                          #  23    0x117214  1      OPC=nop             
  nop                          #  24    0x117215  1      OPC=nop             
  nop                          #  25    0x117216  1      OPC=nop             
  nop                          #  26    0x117217  1      OPC=nop             
  nop                          #  27    0x117218  1      OPC=nop             
  nop                          #  28    0x117219  1      OPC=nop             
  nop                          #  29    0x11721a  1      OPC=nop             
  nop                          #  30    0x11721b  1      OPC=nop             
  nop                          #  31    0x11721c  1      OPC=nop             
  nop                          #  32    0x11721d  1      OPC=nop             
  nop                          #  33    0x11721e  1      OPC=nop             
  nop                          #  34    0x11721f  1      OPC=nop             
  nop                          #  35    0x117220  1      OPC=nop             
  nop                          #  36    0x117221  1      OPC=nop             
  nop                          #  37    0x117222  1      OPC=nop             
  nop                          #  38    0x117223  1      OPC=nop             
  nop                          #  39    0x117224  1      OPC=nop             
  nop                          #  40    0x117225  1      OPC=nop             
.L_117220:                     #        0x117226  0      OPC=<label>         
  leal -0x8(%r12), %r14d       #  41    0x117226  5      OPC=leal_r32_m16    
  movl $0x1, %edx              #  42    0x11722b  5      OPC=movl_r32_imm32  
  movl %r14d, %edi             #  43    0x117230  3      OPC=movl_r32_r32    
  nop                          #  44    0x117233  1      OPC=nop             
  nop                          #  45    0x117234  1      OPC=nop             
  nop                          #  46    0x117235  1      OPC=nop             
  nop                          #  47    0x117236  1      OPC=nop             
  nop                          #  48    0x117237  1      OPC=nop             
  nop                          #  49    0x117238  1      OPC=nop             
  nop                          #  50    0x117239  1      OPC=nop             
  nop                          #  51    0x11723a  1      OPC=nop             
  nop                          #  52    0x11723b  1      OPC=nop             
  nop                          #  53    0x11723c  1      OPC=nop             
  nop                          #  54    0x11723d  1      OPC=nop             
  nop                          #  55    0x11723e  1      OPC=nop             
  nop                          #  56    0x11723f  1      OPC=nop             
  nop                          #  57    0x117240  1      OPC=nop             
  callq .T_276                 #  58    0x117241  5      OPC=callq_label     
  testb $0x2, 0xff61af5(%rip)  #  59    0x117246  7      OPC=testb_m8_imm8   
  movl %eax, %r13d             #  60    0x11724d  3      OPC=movl_r32_r32    
  jne .L_117340                #  61    0x117250  6      OPC=jne_label_1     
  nop                          #  62    0x117256  1      OPC=nop             
  nop                          #  63    0x117257  1      OPC=nop             
  nop                          #  64    0x117258  1      OPC=nop             
  nop                          #  65    0x117259  1      OPC=nop             
  nop                          #  66    0x11725a  1      OPC=nop             
  nop                          #  67    0x11725b  1      OPC=nop             
  nop                          #  68    0x11725c  1      OPC=nop             
  nop                          #  69    0x11725d  1      OPC=nop             
  nop                          #  70    0x11725e  1      OPC=nop             
  nop                          #  71    0x11725f  1      OPC=nop             
  nop                          #  72    0x117260  1      OPC=nop             
  nop                          #  73    0x117261  1      OPC=nop             
  nop                          #  74    0x117262  1      OPC=nop             
  nop                          #  75    0x117263  1      OPC=nop             
  nop                          #  76    0x117264  1      OPC=nop             
  nop                          #  77    0x117265  1      OPC=nop             
.L_117260:                     #        0x117266  0      OPC=<label>         
  testq %r13, %r13             #  78    0x117266  3      OPC=testq_r64_r64   
  je .L_117360                 #  79    0x117269  6      OPC=je_label_1      
  addl $0x8, %r13d             #  80    0x11726f  4      OPC=addl_r32_imm8   
  nop                          #  81    0x117273  1      OPC=nop             
  nop                          #  82    0x117274  1      OPC=nop             
  nop                          #  83    0x117275  1      OPC=nop             
  nop                          #  84    0x117276  1      OPC=nop             
  nop                          #  85    0x117277  1      OPC=nop             
  nop                          #  86    0x117278  1      OPC=nop             
  nop                          #  87    0x117279  1      OPC=nop             
  nop                          #  88    0x11727a  1      OPC=nop             
  nop                          #  89    0x11727b  1      OPC=nop             
  nop                          #  90    0x11727c  1      OPC=nop             
  nop                          #  91    0x11727d  1      OPC=nop             
  nop                          #  92    0x11727e  1      OPC=nop             
  nop                          #  93    0x11727f  1      OPC=nop             
  nop                          #  94    0x117280  1      OPC=nop             
  nop                          #  95    0x117281  1      OPC=nop             
  nop                          #  96    0x117282  1      OPC=nop             
  nop                          #  97    0x117283  1      OPC=nop             
  nop                          #  98    0x117284  1      OPC=nop             
  nop                          #  99    0x117285  1      OPC=nop             
.L_117280:                     #        0x117286  0      OPC=<label>         
  movl %r13d, %eax             #  100   0x117286  3      OPC=movl_r32_r32    
  movq 0x18(%rsp), %rbx        #  101   0x117289  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r12        #  102   0x11728e  5      OPC=movq_r64_m64    
  movq 0x28(%rsp), %r13        #  103   0x117293  5      OPC=movq_r64_m64    
  movq 0x30(%rsp), %r14        #  104   0x117298  5      OPC=movq_r64_m64    
  addl $0x38, %esp             #  105   0x11729d  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  106   0x1172a0  3      OPC=addq_r64_r64    
  popq %r11                    #  107   0x1172a3  2      OPC=popq_r64_1      
  nop                          #  108   0x1172a5  1      OPC=nop             
  andl $0xffffffe0, %r11d      #  109   0x1172a6  7      OPC=andl_r32_imm32  
  nop                          #  110   0x1172ad  1      OPC=nop             
  nop                          #  111   0x1172ae  1      OPC=nop             
  nop                          #  112   0x1172af  1      OPC=nop             
  nop                          #  113   0x1172b0  1      OPC=nop             
  addq %r15, %r11              #  114   0x1172b1  3      OPC=addq_r64_r64    
  jmpq %r11                    #  115   0x1172b4  3      OPC=jmpq_r64        
  nop                          #  116   0x1172b7  1      OPC=nop             
  nop                          #  117   0x1172b8  1      OPC=nop             
  nop                          #  118   0x1172b9  1      OPC=nop             
  nop                          #  119   0x1172ba  1      OPC=nop             
  nop                          #  120   0x1172bb  1      OPC=nop             
  nop                          #  121   0x1172bc  1      OPC=nop             
  nop                          #  122   0x1172bd  1      OPC=nop             
  nop                          #  123   0x1172be  1      OPC=nop             
  nop                          #  124   0x1172bf  1      OPC=nop             
  nop                          #  125   0x1172c0  1      OPC=nop             
  nop                          #  126   0x1172c1  1      OPC=nop             
  nop                          #  127   0x1172c2  1      OPC=nop             
  nop                          #  128   0x1172c3  1      OPC=nop             
  nop                          #  129   0x1172c4  1      OPC=nop             
  nop                          #  130   0x1172c5  1      OPC=nop             
  nop                          #  131   0x1172c6  1      OPC=nop             
  nop                          #  132   0x1172c7  1      OPC=nop             
  nop                          #  133   0x1172c8  1      OPC=nop             
  nop                          #  134   0x1172c9  1      OPC=nop             
  nop                          #  135   0x1172ca  1      OPC=nop             
  nop                          #  136   0x1172cb  1      OPC=nop             
  nop                          #  137   0x1172cc  1      OPC=nop             
.L_1172c0:                     #        0x1172cd  0      OPC=<label>         
  movl $0x1, %eax              #  138   0x1172cd  5      OPC=movl_r32_imm32  
  xchgl %eax, 0xff61a75(%rip)  #  139   0x1172d2  6      OPC=xchgl_m32_r32   
  testl %eax, %eax             #  140   0x1172d8  2      OPC=testl_r32_r32   
  je .L_117220                 #  141   0x1172da  6      OPC=je_label_1      
  movl $0x10078d40, %edi       #  142   0x1172e0  5      OPC=movl_r32_imm32  
  movl %esi, 0x8(%rsp)         #  143   0x1172e5  4      OPC=movl_m32_r32    
  xorl %r13d, %r13d            #  144   0x1172e9  3      OPC=xorl_r32_r32    
  nop                          #  145   0x1172ec  1      OPC=nop             
  nop                          #  146   0x1172ed  1      OPC=nop             
  nop                          #  147   0x1172ee  1      OPC=nop             
  nop                          #  148   0x1172ef  1      OPC=nop             
  nop                          #  149   0x1172f0  1      OPC=nop             
  nop                          #  150   0x1172f1  1      OPC=nop             
  nop                          #  151   0x1172f2  1      OPC=nop             
  nop                          #  152   0x1172f3  1      OPC=nop             
  nop                          #  153   0x1172f4  1      OPC=nop             
  nop                          #  154   0x1172f5  1      OPC=nop             
  nop                          #  155   0x1172f6  1      OPC=nop             
  nop                          #  156   0x1172f7  1      OPC=nop             
  nop                          #  157   0x1172f8  1      OPC=nop             
  nop                          #  158   0x1172f9  1      OPC=nop             
  nop                          #  159   0x1172fa  1      OPC=nop             
  nop                          #  160   0x1172fb  1      OPC=nop             
  nop                          #  161   0x1172fc  1      OPC=nop             
  nop                          #  162   0x1172fd  1      OPC=nop             
  nop                          #  163   0x1172fe  1      OPC=nop             
  nop                          #  164   0x1172ff  1      OPC=nop             
  nop                          #  165   0x117300  1      OPC=nop             
  nop                          #  166   0x117301  1      OPC=nop             
  nop                          #  167   0x117302  1      OPC=nop             
  nop                          #  168   0x117303  1      OPC=nop             
  nop                          #  169   0x117304  1      OPC=nop             
  nop                          #  170   0x117305  1      OPC=nop             
  nop                          #  171   0x117306  1      OPC=nop             
  nop                          #  172   0x117307  1      OPC=nop             
  callq .spin_acquire_lock     #  173   0x117308  5      OPC=callq_label     
  testl %eax, %eax             #  174   0x11730d  2      OPC=testl_r32_r32   
  movl 0x8(%rsp), %esi         #  175   0x11730f  4      OPC=movl_r32_m32    
  je .L_117220                 #  176   0x117313  6      OPC=je_label_1      
  jmpq .L_117280               #  177   0x117319  5      OPC=jmpq_label_1    
  nop                          #  178   0x11731e  1      OPC=nop             
  nop                          #  179   0x11731f  1      OPC=nop             
  nop                          #  180   0x117320  1      OPC=nop             
  nop                          #  181   0x117321  1      OPC=nop             
  nop                          #  182   0x117322  1      OPC=nop             
  nop                          #  183   0x117323  1      OPC=nop             
  nop                          #  184   0x117324  1      OPC=nop             
  nop                          #  185   0x117325  1      OPC=nop             
  nop                          #  186   0x117326  1      OPC=nop             
  nop                          #  187   0x117327  1      OPC=nop             
  nop                          #  188   0x117328  1      OPC=nop             
  nop                          #  189   0x117329  1      OPC=nop             
  nop                          #  190   0x11732a  1      OPC=nop             
  nop                          #  191   0x11732b  1      OPC=nop             
  nop                          #  192   0x11732c  1      OPC=nop             
.L_117320:                     #        0x11732d  0      OPC=<label>         
  leal 0xb(%rbx), %esi         #  193   0x11732d  3      OPC=leal_r32_m16    
  andl $0xfffffff8, %esi       #  194   0x117330  6      OPC=andl_r32_imm32  
  nop                          #  195   0x117336  1      OPC=nop             
  nop                          #  196   0x117337  1      OPC=nop             
  nop                          #  197   0x117338  1      OPC=nop             
  testb $0x2, 0xff61a0f(%rip)  #  198   0x117339  7      OPC=testb_m8_imm8   
  je .L_117220                 #  199   0x117340  6      OPC=je_label_1      
  jmpq .L_1172c0               #  200   0x117346  2      OPC=jmpq_label      
  nop                          #  201   0x117348  1      OPC=nop             
  nop                          #  202   0x117349  1      OPC=nop             
  nop                          #  203   0x11734a  1      OPC=nop             
  nop                          #  204   0x11734b  1      OPC=nop             
  nop                          #  205   0x11734c  1      OPC=nop             
  nop                          #  206   0x11734d  1      OPC=nop             
  nop                          #  207   0x11734e  1      OPC=nop             
  nop                          #  208   0x11734f  1      OPC=nop             
  nop                          #  209   0x117350  1      OPC=nop             
  nop                          #  210   0x117351  1      OPC=nop             
  nop                          #  211   0x117352  1      OPC=nop             
.L_117340:                     #        0x117353  0      OPC=<label>         
  mfence                       #  212   0x117353  3      OPC=mfence          
  movl $0x0, 0xff619f3(%rip)   #  213   0x117356  10     OPC=movl_m32_imm32  
  jmpq .L_117260               #  214   0x117360  5      OPC=jmpq_label_1    
  nop                          #  215   0x117365  1      OPC=nop             
  nop                          #  216   0x117366  1      OPC=nop             
  nop                          #  217   0x117367  1      OPC=nop             
  nop                          #  218   0x117368  1      OPC=nop             
  nop                          #  219   0x117369  1      OPC=nop             
  nop                          #  220   0x11736a  1      OPC=nop             
  nop                          #  221   0x11736b  1      OPC=nop             
  nop                          #  222   0x11736c  1      OPC=nop             
  nop                          #  223   0x11736d  1      OPC=nop             
  nop                          #  224   0x11736e  1      OPC=nop             
  nop                          #  225   0x11736f  1      OPC=nop             
  nop                          #  226   0x117370  1      OPC=nop             
  nop                          #  227   0x117371  1      OPC=nop             
  nop                          #  228   0x117372  1      OPC=nop             
.L_117360:                     #        0x117373  0      OPC=<label>         
  movl %ebx, %edi              #  229   0x117373  2      OPC=movl_r32_r32    
  nop                          #  230   0x117375  1      OPC=nop             
  nop                          #  231   0x117376  1      OPC=nop             
  nop                          #  232   0x117377  1      OPC=nop             
  nop                          #  233   0x117378  1      OPC=nop             
  nop                          #  234   0x117379  1      OPC=nop             
  nop                          #  235   0x11737a  1      OPC=nop             
  nop                          #  236   0x11737b  1      OPC=nop             
  nop                          #  237   0x11737c  1      OPC=nop             
  nop                          #  238   0x11737d  1      OPC=nop             
  nop                          #  239   0x11737e  1      OPC=nop             
  nop                          #  240   0x11737f  1      OPC=nop             
  nop                          #  241   0x117380  1      OPC=nop             
  nop                          #  242   0x117381  1      OPC=nop             
  nop                          #  243   0x117382  1      OPC=nop             
  nop                          #  244   0x117383  1      OPC=nop             
  nop                          #  245   0x117384  1      OPC=nop             
  nop                          #  246   0x117385  1      OPC=nop             
  nop                          #  247   0x117386  1      OPC=nop             
  nop                          #  248   0x117387  1      OPC=nop             
  nop                          #  249   0x117388  1      OPC=nop             
  nop                          #  250   0x117389  1      OPC=nop             
  nop                          #  251   0x11738a  1      OPC=nop             
  nop                          #  252   0x11738b  1      OPC=nop             
  nop                          #  253   0x11738c  1      OPC=nop             
  nop                          #  254   0x11738d  1      OPC=nop             
  callq .malloc                #  255   0x11738e  5      OPC=callq_label     
  movl %eax, %r13d             #  256   0x117393  3      OPC=movl_r32_r32    
  testq %r13, %r13             #  257   0x117396  3      OPC=testq_r64_r64   
  je .L_117280                 #  258   0x117399  6      OPC=je_label_1      
  movl %r14d, %r14d            #  259   0x11739f  3      OPC=movl_r32_r32    
  movl 0x4(%r15,%r14,1), %edx  #  260   0x1173a2  5      OPC=movl_r32_m32    
  movl %r13d, %edi             #  261   0x1173a7  3      OPC=movl_r32_r32    
  movl %r12d, %esi             #  262   0x1173aa  3      OPC=movl_r32_r32    
  movl %edx, %eax              #  263   0x1173ad  2      OPC=movl_r32_r32    
  andl $0x3, %eax              #  264   0x1173af  3      OPC=andl_r32_imm8   
  nop                          #  265   0x1173b2  1      OPC=nop             
  cmpl $0x1, %eax              #  266   0x1173b3  3      OPC=cmpl_r32_imm8   
  sbbl %eax, %eax              #  267   0x1173b6  2      OPC=sbbl_r32_r32    
  andl $0xfffffff8, %edx       #  268   0x1173b8  6      OPC=andl_r32_imm32  
  nop                          #  269   0x1173be  1      OPC=nop             
  nop                          #  270   0x1173bf  1      OPC=nop             
  nop                          #  271   0x1173c0  1      OPC=nop             
  andl $0x4, %eax              #  272   0x1173c1  3      OPC=andl_r32_imm8   
  addl $0x4, %eax              #  273   0x1173c4  3      OPC=addl_r32_imm8   
  subl %eax, %edx              #  274   0x1173c7  2      OPC=subl_r32_r32    
  cmpl %ebx, %edx              #  275   0x1173c9  2      OPC=cmpl_r32_r32    
  cmoval %ebx, %edx            #  276   0x1173cb  3      OPC=cmoval_r32_r32  
  nop                          #  277   0x1173ce  1      OPC=nop             
  nop                          #  278   0x1173cf  1      OPC=nop             
  nop                          #  279   0x1173d0  1      OPC=nop             
  nop                          #  280   0x1173d1  1      OPC=nop             
  nop                          #  281   0x1173d2  1      OPC=nop             
  nop                          #  282   0x1173d3  1      OPC=nop             
  callq .memcpy                #  283   0x1173d4  5      OPC=callq_label     
  movl %r12d, %edi             #  284   0x1173d9  3      OPC=movl_r32_r32    
  nop                          #  285   0x1173dc  1      OPC=nop             
  nop                          #  286   0x1173dd  1      OPC=nop             
  nop                          #  287   0x1173de  1      OPC=nop             
  nop                          #  288   0x1173df  1      OPC=nop             
  nop                          #  289   0x1173e0  1      OPC=nop             
  nop                          #  290   0x1173e1  1      OPC=nop             
  nop                          #  291   0x1173e2  1      OPC=nop             
  nop                          #  292   0x1173e3  1      OPC=nop             
  nop                          #  293   0x1173e4  1      OPC=nop             
  nop                          #  294   0x1173e5  1      OPC=nop             
  nop                          #  295   0x1173e6  1      OPC=nop             
  nop                          #  296   0x1173e7  1      OPC=nop             
  nop                          #  297   0x1173e8  1      OPC=nop             
  nop                          #  298   0x1173e9  1      OPC=nop             
  nop                          #  299   0x1173ea  1      OPC=nop             
  nop                          #  300   0x1173eb  1      OPC=nop             
  nop                          #  301   0x1173ec  1      OPC=nop             
  nop                          #  302   0x1173ed  1      OPC=nop             
  nop                          #  303   0x1173ee  1      OPC=nop             
  nop                          #  304   0x1173ef  1      OPC=nop             
  nop                          #  305   0x1173f0  1      OPC=nop             
  nop                          #  306   0x1173f1  1      OPC=nop             
  nop                          #  307   0x1173f2  1      OPC=nop             
  nop                          #  308   0x1173f3  1      OPC=nop             
  callq .free                  #  309   0x1173f4  5      OPC=callq_label     
  jmpq .L_117280               #  310   0x1173f9  5      OPC=jmpq_label_1    
  nop                          #  311   0x1173fe  1      OPC=nop             
  nop                          #  312   0x1173ff  1      OPC=nop             
  nop                          #  313   0x117400  1      OPC=nop             
  nop                          #  314   0x117401  1      OPC=nop             
  nop                          #  315   0x117402  1      OPC=nop             
  nop                          #  316   0x117403  1      OPC=nop             
  nop                          #  317   0x117404  1      OPC=nop             
  nop                          #  318   0x117405  1      OPC=nop             
  nop                          #  319   0x117406  1      OPC=nop             
  nop                          #  320   0x117407  1      OPC=nop             
  nop                          #  321   0x117408  1      OPC=nop             
  nop                          #  322   0x117409  1      OPC=nop             
  nop                          #  323   0x11740a  1      OPC=nop             
  nop                          #  324   0x11740b  1      OPC=nop             
  nop                          #  325   0x11740c  1      OPC=nop             
  nop                          #  326   0x11740d  1      OPC=nop             
  nop                          #  327   0x11740e  1      OPC=nop             
  nop                          #  328   0x11740f  1      OPC=nop             
  nop                          #  329   0x117410  1      OPC=nop             
  nop                          #  330   0x117411  1      OPC=nop             
  nop                          #  331   0x117412  1      OPC=nop             
  nop                          #  332   0x117413  1      OPC=nop             
  nop                          #  333   0x117414  1      OPC=nop             
  nop                          #  334   0x117415  1      OPC=nop             
  nop                          #  335   0x117416  1      OPC=nop             
  nop                          #  336   0x117417  1      OPC=nop             
  nop                          #  337   0x117418  1      OPC=nop             
.L_117400:                     #        0x117419  0      OPC=<label>         
  movq 0x18(%rsp), %rbx        #  338   0x117419  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r12        #  339   0x11741e  5      OPC=movq_r64_m64    
  movl %esi, %edi              #  340   0x117423  2      OPC=movl_r32_r32    
  movq 0x28(%rsp), %r13        #  341   0x117425  5      OPC=movq_r64_m64    
  movq 0x30(%rsp), %r14        #  342   0x11742a  5      OPC=movq_r64_m64    
  addl $0x38, %esp             #  343   0x11742f  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  344   0x117432  3      OPC=addq_r64_r64    
  nop                          #  345   0x117435  1      OPC=nop             
  nop                          #  346   0x117436  1      OPC=nop             
  nop                          #  347   0x117437  1      OPC=nop             
  nop                          #  348   0x117438  1      OPC=nop             
  jmpq .malloc                 #  349   0x117439  5      OPC=jmpq_label_1    
  nop                          #  350   0x11743e  1      OPC=nop             
  nop                          #  351   0x11743f  1      OPC=nop             
  nop                          #  352   0x117440  1      OPC=nop             
  nop                          #  353   0x117441  1      OPC=nop             
  nop                          #  354   0x117442  1      OPC=nop             
  nop                          #  355   0x117443  1      OPC=nop             
  nop                          #  356   0x117444  1      OPC=nop             
  nop                          #  357   0x117445  1      OPC=nop             
  nop                          #  358   0x117446  1      OPC=nop             
  nop                          #  359   0x117447  1      OPC=nop             
  nop                          #  360   0x117448  1      OPC=nop             
  nop                          #  361   0x117449  1      OPC=nop             
  nop                          #  362   0x11744a  1      OPC=nop             
  nop                          #  363   0x11744b  1      OPC=nop             
  nop                          #  364   0x11744c  1      OPC=nop             
  nop                          #  365   0x11744d  1      OPC=nop             
  nop                          #  366   0x11744e  1      OPC=nop             
  nop                          #  367   0x11744f  1      OPC=nop             
  nop                          #  368   0x117450  1      OPC=nop             
  nop                          #  369   0x117451  1      OPC=nop             
  nop                          #  370   0x117452  1      OPC=nop             
  nop                          #  371   0x117453  1      OPC=nop             
  nop                          #  372   0x117454  1      OPC=nop             
  nop                          #  373   0x117455  1      OPC=nop             
  nop                          #  374   0x117456  1      OPC=nop             
  nop                          #  375   0x117457  1      OPC=nop             
  nop                          #  376   0x117458  1      OPC=nop             
.L_117440:                     #        0x117459  0      OPC=<label>         
  nop                          #  377   0x117459  1      OPC=nop             
  nop                          #  378   0x11745a  1      OPC=nop             
  nop                          #  379   0x11745b  1      OPC=nop             
  nop                          #  380   0x11745c  1      OPC=nop             
  nop                          #  381   0x11745d  1      OPC=nop             
  nop                          #  382   0x11745e  1      OPC=nop             
  nop                          #  383   0x11745f  1      OPC=nop             
  nop                          #  384   0x117460  1      OPC=nop             
  nop                          #  385   0x117461  1      OPC=nop             
  nop                          #  386   0x117462  1      OPC=nop             
  nop                          #  387   0x117463  1      OPC=nop             
  nop                          #  388   0x117464  1      OPC=nop             
  nop                          #  389   0x117465  1      OPC=nop             
  nop                          #  390   0x117466  1      OPC=nop             
  nop                          #  391   0x117467  1      OPC=nop             
  nop                          #  392   0x117468  1      OPC=nop             
  nop                          #  393   0x117469  1      OPC=nop             
  nop                          #  394   0x11746a  1      OPC=nop             
  nop                          #  395   0x11746b  1      OPC=nop             
  nop                          #  396   0x11746c  1      OPC=nop             
  nop                          #  397   0x11746d  1      OPC=nop             
  nop                          #  398   0x11746e  1      OPC=nop             
  nop                          #  399   0x11746f  1      OPC=nop             
  nop                          #  400   0x117470  1      OPC=nop             
  nop                          #  401   0x117471  1      OPC=nop             
  nop                          #  402   0x117472  1      OPC=nop             
  nop                          #  403   0x117473  1      OPC=nop             
  callq .__errno               #  404   0x117474  5      OPC=callq_label     
  movl %eax, %eax              #  405   0x117479  2      OPC=movl_r32_r32    
  xorl %r13d, %r13d            #  406   0x11747b  3      OPC=xorl_r32_r32    
  movl %eax, %eax              #  407   0x11747e  2      OPC=movl_r32_r32    
  movl $0xc, (%r15,%rax,1)     #  408   0x117480  8      OPC=movl_m32_imm32  
  jmpq .L_117280               #  409   0x117488  5      OPC=jmpq_label_1    
  nop                          #  410   0x11748d  1      OPC=nop             
  nop                          #  411   0x11748e  1      OPC=nop             
  nop                          #  412   0x11748f  1      OPC=nop             
  nop                          #  413   0x117490  1      OPC=nop             
  nop                          #  414   0x117491  1      OPC=nop             
  nop                          #  415   0x117492  1      OPC=nop             
  nop                          #  416   0x117493  1      OPC=nop             
  nop                          #  417   0x117494  1      OPC=nop             
  nop                          #  418   0x117495  1      OPC=nop             
  nop                          #  419   0x117496  1      OPC=nop             
  nop                          #  420   0x117497  1      OPC=nop             
  nop                          #  421   0x117498  1      OPC=nop             
                                                                             
.size realloc, .-realloc

