  .text
  .globl d_exprlist
  .type d_exprlist, @function

#! file-offset 0x1426a0
#! rip-offset  0x1026a0
#! capacity    384 bytes

# Text                         #  Line  RIP       Bytes  Opcode              
.d_exprlist:                   #        0x1026a0  0      OPC=<label>         
  pushq %r12                   #  1     0x1026a0  2      OPC=pushq_r64_1     
  pushq %rbx                   #  2     0x1026a2  1      OPC=pushq_r64_1     
  movl %edi, %ebx              #  3     0x1026a3  2      OPC=movl_r32_r32    
  subl $0x18, %esp             #  4     0x1026a5  3      OPC=subl_r32_imm8   
  addq %r15, %rsp              #  5     0x1026a8  3      OPC=addq_r64_r64    
  movl %ebx, %ebx              #  6     0x1026ab  2      OPC=movl_r32_r32    
  movl 0xc(%r15,%rbx,1), %eax  #  7     0x1026ad  5      OPC=movl_r32_m32    
  movl $0x0, 0xc(%rsp)         #  8     0x1026b2  8      OPC=movl_m32_imm32  
  nop                          #  9     0x1026ba  1      OPC=nop             
  nop                          #  10    0x1026bb  1      OPC=nop             
  nop                          #  11    0x1026bc  1      OPC=nop             
  nop                          #  12    0x1026bd  1      OPC=nop             
  nop                          #  13    0x1026be  1      OPC=nop             
  nop                          #  14    0x1026bf  1      OPC=nop             
  movl %eax, %eax              #  15    0x1026c0  2      OPC=movl_r32_r32    
  cmpb $0x45, (%r15,%rax,1)    #  16    0x1026c2  5      OPC=cmpb_m8_imm8    
  jne .L_102720                #  17    0x1026c7  2      OPC=jne_label       
  addl $0x1, %eax              #  18    0x1026c9  3      OPC=addl_r32_imm8   
  movl %ebx, %edi              #  19    0x1026cc  2      OPC=movl_r32_r32    
  xorl %ecx, %ecx              #  20    0x1026ce  2      OPC=xorl_r32_r32    
  movl %ebx, %ebx              #  21    0x1026d0  2      OPC=movl_r32_r32    
  movl %eax, 0xc(%r15,%rbx,1)  #  22    0x1026d2  5      OPC=movl_m32_r32    
  xorl %edx, %edx              #  23    0x1026d7  2      OPC=xorl_r32_r32    
  movl $0x29, %esi             #  24    0x1026d9  5      OPC=movl_r32_imm32  
  xchgw %ax, %ax               #  25    0x1026de  2      OPC=xchgw_ax_r16    
  nop                          #  26    0x1026e0  1      OPC=nop             
  nop                          #  27    0x1026e1  1      OPC=nop             
  nop                          #  28    0x1026e2  1      OPC=nop             
  nop                          #  29    0x1026e3  1      OPC=nop             
  nop                          #  30    0x1026e4  1      OPC=nop             
  nop                          #  31    0x1026e5  1      OPC=nop             
  nop                          #  32    0x1026e6  1      OPC=nop             
  nop                          #  33    0x1026e7  1      OPC=nop             
  nop                          #  34    0x1026e8  1      OPC=nop             
  nop                          #  35    0x1026e9  1      OPC=nop             
  nop                          #  36    0x1026ea  1      OPC=nop             
  nop                          #  37    0x1026eb  1      OPC=nop             
  nop                          #  38    0x1026ec  1      OPC=nop             
  nop                          #  39    0x1026ed  1      OPC=nop             
  nop                          #  40    0x1026ee  1      OPC=nop             
  nop                          #  41    0x1026ef  1      OPC=nop             
  nop                          #  42    0x1026f0  1      OPC=nop             
  nop                          #  43    0x1026f1  1      OPC=nop             
  nop                          #  44    0x1026f2  1      OPC=nop             
  nop                          #  45    0x1026f3  1      OPC=nop             
  nop                          #  46    0x1026f4  1      OPC=nop             
  nop                          #  47    0x1026f5  1      OPC=nop             
  nop                          #  48    0x1026f6  1      OPC=nop             
  nop                          #  49    0x1026f7  1      OPC=nop             
  nop                          #  50    0x1026f8  1      OPC=nop             
  nop                          #  51    0x1026f9  1      OPC=nop             
  nop                          #  52    0x1026fa  1      OPC=nop             
  callq .d_make_comp           #  53    0x1026fb  5      OPC=callq_label     
  addl $0x18, %esp             #  54    0x102700  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  55    0x102703  3      OPC=addq_r64_r64    
  movl %eax, %eax              #  56    0x102706  2      OPC=movl_r32_r32    
  popq %rbx                    #  57    0x102708  1      OPC=popq_r64_1      
  popq %r12                    #  58    0x102709  2      OPC=popq_r64_1      
  popq %r11                    #  59    0x10270b  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  60    0x10270d  7      OPC=andl_r32_imm32  
  nop                          #  61    0x102714  1      OPC=nop             
  nop                          #  62    0x102715  1      OPC=nop             
  nop                          #  63    0x102716  1      OPC=nop             
  nop                          #  64    0x102717  1      OPC=nop             
  addq %r15, %r11              #  65    0x102718  3      OPC=addq_r64_r64    
  jmpq %r11                    #  66    0x10271b  3      OPC=jmpq_r64        
  nop                          #  67    0x10271e  1      OPC=nop             
  nop                          #  68    0x10271f  1      OPC=nop             
  nop                          #  69    0x102720  1      OPC=nop             
  nop                          #  70    0x102721  1      OPC=nop             
  nop                          #  71    0x102722  1      OPC=nop             
  nop                          #  72    0x102723  1      OPC=nop             
  nop                          #  73    0x102724  1      OPC=nop             
  nop                          #  74    0x102725  1      OPC=nop             
  nop                          #  75    0x102726  1      OPC=nop             
.L_102720:                     #        0x102727  0      OPC=<label>         
  leal 0xc(%rsp), %r12d        #  76    0x102727  5      OPC=leal_r32_m16    
  jmpq .L_102760               #  77    0x10272c  2      OPC=jmpq_label      
  nop                          #  78    0x10272e  1      OPC=nop             
  nop                          #  79    0x10272f  1      OPC=nop             
  nop                          #  80    0x102730  1      OPC=nop             
  nop                          #  81    0x102731  1      OPC=nop             
  nop                          #  82    0x102732  1      OPC=nop             
  nop                          #  83    0x102733  1      OPC=nop             
  nop                          #  84    0x102734  1      OPC=nop             
  nop                          #  85    0x102735  1      OPC=nop             
  nop                          #  86    0x102736  1      OPC=nop             
  nop                          #  87    0x102737  1      OPC=nop             
  nop                          #  88    0x102738  1      OPC=nop             
  nop                          #  89    0x102739  1      OPC=nop             
  nop                          #  90    0x10273a  1      OPC=nop             
  nop                          #  91    0x10273b  1      OPC=nop             
  nop                          #  92    0x10273c  1      OPC=nop             
  nop                          #  93    0x10273d  1      OPC=nop             
  nop                          #  94    0x10273e  1      OPC=nop             
  nop                          #  95    0x10273f  1      OPC=nop             
  nop                          #  96    0x102740  1      OPC=nop             
  nop                          #  97    0x102741  1      OPC=nop             
  nop                          #  98    0x102742  1      OPC=nop             
  nop                          #  99    0x102743  1      OPC=nop             
  nop                          #  100   0x102744  1      OPC=nop             
  nop                          #  101   0x102745  1      OPC=nop             
  nop                          #  102   0x102746  1      OPC=nop             
.L_102740:                     #        0x102747  0      OPC=<label>         
  leal 0x8(%rax), %r12d        #  103   0x102747  4      OPC=leal_r32_m16    
  nop                          #  104   0x10274b  1      OPC=nop             
  nop                          #  105   0x10274c  1      OPC=nop             
  nop                          #  106   0x10274d  1      OPC=nop             
  nop                          #  107   0x10274e  1      OPC=nop             
  nop                          #  108   0x10274f  1      OPC=nop             
  nop                          #  109   0x102750  1      OPC=nop             
  nop                          #  110   0x102751  1      OPC=nop             
  nop                          #  111   0x102752  1      OPC=nop             
  nop                          #  112   0x102753  1      OPC=nop             
  nop                          #  113   0x102754  1      OPC=nop             
  nop                          #  114   0x102755  1      OPC=nop             
  nop                          #  115   0x102756  1      OPC=nop             
  nop                          #  116   0x102757  1      OPC=nop             
  nop                          #  117   0x102758  1      OPC=nop             
  nop                          #  118   0x102759  1      OPC=nop             
  nop                          #  119   0x10275a  1      OPC=nop             
  nop                          #  120   0x10275b  1      OPC=nop             
  nop                          #  121   0x10275c  1      OPC=nop             
  nop                          #  122   0x10275d  1      OPC=nop             
  nop                          #  123   0x10275e  1      OPC=nop             
  nop                          #  124   0x10275f  1      OPC=nop             
  nop                          #  125   0x102760  1      OPC=nop             
  nop                          #  126   0x102761  1      OPC=nop             
  nop                          #  127   0x102762  1      OPC=nop             
  nop                          #  128   0x102763  1      OPC=nop             
  nop                          #  129   0x102764  1      OPC=nop             
  nop                          #  130   0x102765  1      OPC=nop             
  nop                          #  131   0x102766  1      OPC=nop             
.L_102760:                     #        0x102767  0      OPC=<label>         
  movl %ebx, %edi              #  132   0x102767  2      OPC=movl_r32_r32    
  nop                          #  133   0x102769  1      OPC=nop             
  nop                          #  134   0x10276a  1      OPC=nop             
  nop                          #  135   0x10276b  1      OPC=nop             
  nop                          #  136   0x10276c  1      OPC=nop             
  nop                          #  137   0x10276d  1      OPC=nop             
  nop                          #  138   0x10276e  1      OPC=nop             
  nop                          #  139   0x10276f  1      OPC=nop             
  nop                          #  140   0x102770  1      OPC=nop             
  nop                          #  141   0x102771  1      OPC=nop             
  nop                          #  142   0x102772  1      OPC=nop             
  nop                          #  143   0x102773  1      OPC=nop             
  nop                          #  144   0x102774  1      OPC=nop             
  nop                          #  145   0x102775  1      OPC=nop             
  nop                          #  146   0x102776  1      OPC=nop             
  nop                          #  147   0x102777  1      OPC=nop             
  nop                          #  148   0x102778  1      OPC=nop             
  nop                          #  149   0x102779  1      OPC=nop             
  nop                          #  150   0x10277a  1      OPC=nop             
  nop                          #  151   0x10277b  1      OPC=nop             
  nop                          #  152   0x10277c  1      OPC=nop             
  nop                          #  153   0x10277d  1      OPC=nop             
  nop                          #  154   0x10277e  1      OPC=nop             
  nop                          #  155   0x10277f  1      OPC=nop             
  nop                          #  156   0x102780  1      OPC=nop             
  nop                          #  157   0x102781  1      OPC=nop             
  callq .d_expression          #  158   0x102782  5      OPC=callq_label     
  movl %eax, %edx              #  159   0x102787  2      OPC=movl_r32_r32    
  testq %rdx, %rdx             #  160   0x102789  3      OPC=testq_r64_r64   
  je .L_102800                 #  161   0x10278c  2      OPC=je_label        
  xorl %ecx, %ecx              #  162   0x10278e  2      OPC=xorl_r32_r32    
  movl $0x29, %esi             #  163   0x102790  5      OPC=movl_r32_imm32  
  movl %ebx, %edi              #  164   0x102795  2      OPC=movl_r32_r32    
  nop                          #  165   0x102797  1      OPC=nop             
  nop                          #  166   0x102798  1      OPC=nop             
  nop                          #  167   0x102799  1      OPC=nop             
  nop                          #  168   0x10279a  1      OPC=nop             
  nop                          #  169   0x10279b  1      OPC=nop             
  nop                          #  170   0x10279c  1      OPC=nop             
  nop                          #  171   0x10279d  1      OPC=nop             
  nop                          #  172   0x10279e  1      OPC=nop             
  nop                          #  173   0x10279f  1      OPC=nop             
  nop                          #  174   0x1027a0  1      OPC=nop             
  nop                          #  175   0x1027a1  1      OPC=nop             
  callq .d_make_comp           #  176   0x1027a2  5      OPC=callq_label     
  movl %eax, %eax              #  177   0x1027a7  2      OPC=movl_r32_r32    
  testq %rax, %rax             #  178   0x1027a9  3      OPC=testq_r64_r64   
  movl %r12d, %r12d            #  179   0x1027ac  3      OPC=movl_r32_r32    
  movl %eax, (%r15,%r12,1)     #  180   0x1027af  4      OPC=movl_m32_r32    
  je .L_102800                 #  181   0x1027b3  2      OPC=je_label        
  movl %ebx, %ebx              #  182   0x1027b5  2      OPC=movl_r32_r32    
  movl 0xc(%r15,%rbx,1), %edx  #  183   0x1027b7  5      OPC=movl_r32_m32    
  movl %edx, %edx              #  184   0x1027bc  2      OPC=movl_r32_r32    
  cmpb $0x45, (%r15,%rdx,1)    #  185   0x1027be  5      OPC=cmpb_m8_imm8    
  nop                          #  186   0x1027c3  1      OPC=nop             
  nop                          #  187   0x1027c4  1      OPC=nop             
  nop                          #  188   0x1027c5  1      OPC=nop             
  nop                          #  189   0x1027c6  1      OPC=nop             
  jne .L_102740                #  190   0x1027c7  6      OPC=jne_label_1     
  addl $0x1, %edx              #  191   0x1027cd  3      OPC=addl_r32_imm8   
  movl 0xc(%rsp), %eax         #  192   0x1027d0  4      OPC=movl_r32_m32    
  movl %ebx, %ebx              #  193   0x1027d4  2      OPC=movl_r32_r32    
  movl %edx, 0xc(%r15,%rbx,1)  #  194   0x1027d6  5      OPC=movl_m32_r32    
  addl $0x18, %esp             #  195   0x1027db  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  196   0x1027de  3      OPC=addq_r64_r64    
  popq %rbx                    #  197   0x1027e1  1      OPC=popq_r64_1      
  popq %r12                    #  198   0x1027e2  2      OPC=popq_r64_1      
  popq %r11                    #  199   0x1027e4  2      OPC=popq_r64_1      
  nop                          #  200   0x1027e6  1      OPC=nop             
  andl $0xffffffe0, %r11d      #  201   0x1027e7  7      OPC=andl_r32_imm32  
  nop                          #  202   0x1027ee  1      OPC=nop             
  nop                          #  203   0x1027ef  1      OPC=nop             
  nop                          #  204   0x1027f0  1      OPC=nop             
  nop                          #  205   0x1027f1  1      OPC=nop             
  addq %r15, %r11              #  206   0x1027f2  3      OPC=addq_r64_r64    
  jmpq %r11                    #  207   0x1027f5  3      OPC=jmpq_r64        
  nop                          #  208   0x1027f8  1      OPC=nop             
  nop                          #  209   0x1027f9  1      OPC=nop             
  nop                          #  210   0x1027fa  1      OPC=nop             
  nop                          #  211   0x1027fb  1      OPC=nop             
  nop                          #  212   0x1027fc  1      OPC=nop             
  nop                          #  213   0x1027fd  1      OPC=nop             
  nop                          #  214   0x1027fe  1      OPC=nop             
  nop                          #  215   0x1027ff  1      OPC=nop             
  nop                          #  216   0x102800  1      OPC=nop             
  nop                          #  217   0x102801  1      OPC=nop             
  nop                          #  218   0x102802  1      OPC=nop             
  nop                          #  219   0x102803  1      OPC=nop             
  nop                          #  220   0x102804  1      OPC=nop             
  nop                          #  221   0x102805  1      OPC=nop             
  nop                          #  222   0x102806  1      OPC=nop             
  nop                          #  223   0x102807  1      OPC=nop             
  nop                          #  224   0x102808  1      OPC=nop             
  nop                          #  225   0x102809  1      OPC=nop             
  nop                          #  226   0x10280a  1      OPC=nop             
  nop                          #  227   0x10280b  1      OPC=nop             
  nop                          #  228   0x10280c  1      OPC=nop             
  nop                          #  229   0x10280d  1      OPC=nop             
.L_102800:                     #        0x10280e  0      OPC=<label>         
  addl $0x18, %esp             #  230   0x10280e  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  231   0x102811  3      OPC=addq_r64_r64    
  xorl %eax, %eax              #  232   0x102814  2      OPC=xorl_r32_r32    
  popq %rbx                    #  233   0x102816  1      OPC=popq_r64_1      
  popq %r12                    #  234   0x102817  2      OPC=popq_r64_1      
  popq %r11                    #  235   0x102819  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d      #  236   0x10281b  7      OPC=andl_r32_imm32  
  nop                          #  237   0x102822  1      OPC=nop             
  nop                          #  238   0x102823  1      OPC=nop             
  nop                          #  239   0x102824  1      OPC=nop             
  nop                          #  240   0x102825  1      OPC=nop             
  addq %r15, %r11              #  241   0x102826  3      OPC=addq_r64_r64    
  jmpq %r11                    #  242   0x102829  3      OPC=jmpq_r64        
  nop                          #  243   0x10282c  1      OPC=nop             
  nop                          #  244   0x10282d  1      OPC=nop             
  nop                          #  245   0x10282e  1      OPC=nop             
  nop                          #  246   0x10282f  1      OPC=nop             
  nop                          #  247   0x102830  1      OPC=nop             
  nop                          #  248   0x102831  1      OPC=nop             
  nop                          #  249   0x102832  1      OPC=nop             
  nop                          #  250   0x102833  1      OPC=nop             
  nop                          #  251   0x102834  1      OPC=nop             
                                                                             
.size d_exprlist, .-d_exprlist

