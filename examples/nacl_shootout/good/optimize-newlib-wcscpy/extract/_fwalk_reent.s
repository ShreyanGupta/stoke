  .text
  .globl _fwalk_reent
  .type _fwalk_reent, @function

#! file-offset 0x15d680
#! rip-offset  0x11d680
#! capacity    448 bytes

# Text                            #  Line  RIP       Bytes  Opcode              
._fwalk_reent:                    #        0x11d680  0      OPC=<label>         
  pushq %r14                      #  1     0x11d680  2      OPC=pushq_r64_1     
  movl %edi, %edi                 #  2     0x11d682  2      OPC=movl_r32_r32    
  movl %esi, %esi                 #  3     0x11d684  2      OPC=movl_r32_r32    
  pushq %r13                      #  4     0x11d686  2      OPC=pushq_r64_1     
  pushq %r12                      #  5     0x11d688  2      OPC=pushq_r64_1     
  pushq %rbx                      #  6     0x11d68a  1      OPC=pushq_r64_1     
  subl $0x18, %esp                #  7     0x11d68b  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                 #  8     0x11d68e  3      OPC=addq_r64_r64    
  movq %rdi, (%rsp)               #  9     0x11d691  4      OPC=movq_m64_r64    
  movq %rsi, 0x8(%rsp)            #  10    0x11d695  5      OPC=movq_m64_r64    
  nop                             #  11    0x11d69a  1      OPC=nop             
  callq .__sfp_lock_acquire       #  12    0x11d69b  5      OPC=callq_label     
  cmpq $0x0, (%rsp)               #  13    0x11d6a0  5      OPC=cmpq_m64_imm8   
  je .L_11d6c0                    #  14    0x11d6a5  2      OPC=je_label        
  movq (%rsp), %rax               #  15    0x11d6a7  4      OPC=movq_r64_m64    
  movl %eax, %eax                 #  16    0x11d6ab  2      OPC=movl_r32_r32    
  movl 0x38(%r15,%rax,1), %edx    #  17    0x11d6ad  5      OPC=movl_r32_m32    
  testl %edx, %edx                #  18    0x11d6b2  2      OPC=testl_r32_r32   
  je .L_11d800                    #  19    0x11d6b4  6      OPC=je_label_1      
  nop                             #  20    0x11d6ba  1      OPC=nop             
  nop                             #  21    0x11d6bb  1      OPC=nop             
  nop                             #  22    0x11d6bc  1      OPC=nop             
  nop                             #  23    0x11d6bd  1      OPC=nop             
  nop                             #  24    0x11d6be  1      OPC=nop             
  nop                             #  25    0x11d6bf  1      OPC=nop             
.L_11d6c0:                        #        0x11d6c0  0      OPC=<label>         
  movl (%rsp), %r13d              #  26    0x11d6c0  4      OPC=movl_r32_m32    
  xorl %r14d, %r14d               #  27    0x11d6c4  3      OPC=xorl_r32_r32    
  addl $0x2e0, %r13d              #  28    0x11d6c7  7      OPC=addl_r32_imm32  
  testq %r13, %r13                #  29    0x11d6ce  3      OPC=testq_r64_r64   
  je .L_11d7c0                    #  30    0x11d6d1  6      OPC=je_label_1      
  nop                             #  31    0x11d6d7  1      OPC=nop             
  nop                             #  32    0x11d6d8  1      OPC=nop             
  nop                             #  33    0x11d6d9  1      OPC=nop             
  nop                             #  34    0x11d6da  1      OPC=nop             
  nop                             #  35    0x11d6db  1      OPC=nop             
  nop                             #  36    0x11d6dc  1      OPC=nop             
  nop                             #  37    0x11d6dd  1      OPC=nop             
  nop                             #  38    0x11d6de  1      OPC=nop             
  nop                             #  39    0x11d6df  1      OPC=nop             
.L_11d6e0:                        #        0x11d6e0  0      OPC=<label>         
  movl %r13d, %r13d               #  40    0x11d6e0  3      OPC=movl_r32_r32    
  movl 0x4(%r15,%r13,1), %r12d    #  41    0x11d6e3  5      OPC=movl_r32_m32    
  movl %r13d, %r13d               #  42    0x11d6e8  3      OPC=movl_r32_r32    
  movl 0x8(%r15,%r13,1), %ebx     #  43    0x11d6eb  5      OPC=movl_r32_m32    
  subl $0x1, %r12d                #  44    0x11d6f0  4      OPC=subl_r32_imm8   
  jns .L_11d720                   #  45    0x11d6f4  2      OPC=jns_label       
  jmpq .L_11d7a0                  #  46    0x11d6f6  5      OPC=jmpq_label_1    
  nop                             #  47    0x11d6fb  1      OPC=nop             
  nop                             #  48    0x11d6fc  1      OPC=nop             
  nop                             #  49    0x11d6fd  1      OPC=nop             
  nop                             #  50    0x11d6fe  1      OPC=nop             
  nop                             #  51    0x11d6ff  1      OPC=nop             
.L_11d700:                        #        0x11d700  0      OPC=<label>         
  subl $0xffffff80, %ebx          #  52    0x11d700  6      OPC=subl_r32_imm32  
  nop                             #  53    0x11d706  1      OPC=nop             
  nop                             #  54    0x11d707  1      OPC=nop             
  nop                             #  55    0x11d708  1      OPC=nop             
  nop                             #  56    0x11d709  1      OPC=nop             
  nop                             #  57    0x11d70a  1      OPC=nop             
  nop                             #  58    0x11d70b  1      OPC=nop             
  nop                             #  59    0x11d70c  1      OPC=nop             
  nop                             #  60    0x11d70d  1      OPC=nop             
  nop                             #  61    0x11d70e  1      OPC=nop             
  nop                             #  62    0x11d70f  1      OPC=nop             
  nop                             #  63    0x11d710  1      OPC=nop             
  nop                             #  64    0x11d711  1      OPC=nop             
  nop                             #  65    0x11d712  1      OPC=nop             
  nop                             #  66    0x11d713  1      OPC=nop             
  nop                             #  67    0x11d714  1      OPC=nop             
  nop                             #  68    0x11d715  1      OPC=nop             
  nop                             #  69    0x11d716  1      OPC=nop             
  nop                             #  70    0x11d717  1      OPC=nop             
  nop                             #  71    0x11d718  1      OPC=nop             
  nop                             #  72    0x11d719  1      OPC=nop             
  nop                             #  73    0x11d71a  1      OPC=nop             
  nop                             #  74    0x11d71b  1      OPC=nop             
  nop                             #  75    0x11d71c  1      OPC=nop             
  nop                             #  76    0x11d71d  1      OPC=nop             
  nop                             #  77    0x11d71e  1      OPC=nop             
  nop                             #  78    0x11d71f  1      OPC=nop             
  nop                             #  79    0x11d720  1      OPC=nop             
  nop                             #  80    0x11d721  1      OPC=nop             
  nop                             #  81    0x11d722  1      OPC=nop             
  nop                             #  82    0x11d723  1      OPC=nop             
  nop                             #  83    0x11d724  1      OPC=nop             
  nop                             #  84    0x11d725  1      OPC=nop             
.L_11d720:                        #        0x11d726  0      OPC=<label>         
  movl %ebx, %ebx                 #  85    0x11d726  2      OPC=movl_r32_r32    
  movzwl 0xc(%r15,%rbx,1), %eax   #  86    0x11d728  6      OPC=movzwl_r32_m16  
  testw %ax, %ax                  #  87    0x11d72e  3      OPC=testw_r16_r16   
  je .L_11d780                    #  88    0x11d731  2      OPC=je_label        
  cmpw $0x1, %ax                  #  89    0x11d733  4      OPC=cmpw_ax_imm16   
  jbe .L_11d780                   #  90    0x11d737  2      OPC=jbe_label       
  movl %ebx, %ebx                 #  91    0x11d739  2      OPC=movl_r32_r32    
  cmpw $0xffff, 0xe(%r15,%rbx,1)  #  92    0x11d73b  8      OPC=cmpw_m16_imm16  
  nop                             #  93    0x11d743  1      OPC=nop             
  nop                             #  94    0x11d744  1      OPC=nop             
  nop                             #  95    0x11d745  1      OPC=nop             
  nop                             #  96    0x11d746  1      OPC=nop             
  nop                             #  97    0x11d747  1      OPC=nop             
  nop                             #  98    0x11d748  1      OPC=nop             
  nop                             #  99    0x11d749  1      OPC=nop             
  nop                             #  100   0x11d74a  1      OPC=nop             
  nop                             #  101   0x11d74b  1      OPC=nop             
  nop                             #  102   0x11d74c  1      OPC=nop             
  nop                             #  103   0x11d74d  1      OPC=nop             
  je .L_11d780                    #  104   0x11d74e  2      OPC=je_label        
  movl %ebx, %esi                 #  105   0x11d750  2      OPC=movl_r32_r32    
  movl (%rsp), %edi               #  106   0x11d752  3      OPC=movl_r32_m32    
  movq 0x8(%rsp), %rdx            #  107   0x11d755  5      OPC=movq_r64_m64    
  nop                             #  108   0x11d75a  1      OPC=nop             
  nop                             #  109   0x11d75b  1      OPC=nop             
  nop                             #  110   0x11d75c  1      OPC=nop             
  nop                             #  111   0x11d75d  1      OPC=nop             
  nop                             #  112   0x11d75e  1      OPC=nop             
  nop                             #  113   0x11d75f  1      OPC=nop             
  nop                             #  114   0x11d760  1      OPC=nop             
  nop                             #  115   0x11d761  1      OPC=nop             
  nop                             #  116   0x11d762  1      OPC=nop             
  nop                             #  117   0x11d763  1      OPC=nop             
  nop                             #  118   0x11d764  1      OPC=nop             
  nop                             #  119   0x11d765  1      OPC=nop             
  andl $0xffffffe0, %edx          #  120   0x11d766  6      OPC=andl_r32_imm32  
  nop                             #  121   0x11d76c  1      OPC=nop             
  nop                             #  122   0x11d76d  1      OPC=nop             
  nop                             #  123   0x11d76e  1      OPC=nop             
  addq %r15, %rdx                 #  124   0x11d76f  3      OPC=addq_r64_r64    
  callq %rdx                      #  125   0x11d772  2      OPC=callq_r64       
  orl %eax, %r14d                 #  126   0x11d774  3      OPC=orl_r32_r32     
  nop                             #  127   0x11d777  1      OPC=nop             
  nop                             #  128   0x11d778  1      OPC=nop             
  nop                             #  129   0x11d779  1      OPC=nop             
  nop                             #  130   0x11d77a  1      OPC=nop             
  nop                             #  131   0x11d77b  1      OPC=nop             
  nop                             #  132   0x11d77c  1      OPC=nop             
  nop                             #  133   0x11d77d  1      OPC=nop             
  nop                             #  134   0x11d77e  1      OPC=nop             
  nop                             #  135   0x11d77f  1      OPC=nop             
  nop                             #  136   0x11d780  1      OPC=nop             
  nop                             #  137   0x11d781  1      OPC=nop             
  nop                             #  138   0x11d782  1      OPC=nop             
  nop                             #  139   0x11d783  1      OPC=nop             
  nop                             #  140   0x11d784  1      OPC=nop             
  nop                             #  141   0x11d785  1      OPC=nop             
  nop                             #  142   0x11d786  1      OPC=nop             
  nop                             #  143   0x11d787  1      OPC=nop             
  nop                             #  144   0x11d788  1      OPC=nop             
  nop                             #  145   0x11d789  1      OPC=nop             
  nop                             #  146   0x11d78a  1      OPC=nop             
  nop                             #  147   0x11d78b  1      OPC=nop             
  nop                             #  148   0x11d78c  1      OPC=nop             
  nop                             #  149   0x11d78d  1      OPC=nop             
  nop                             #  150   0x11d78e  1      OPC=nop             
  nop                             #  151   0x11d78f  1      OPC=nop             
  nop                             #  152   0x11d790  1      OPC=nop             
  nop                             #  153   0x11d791  1      OPC=nop             
  nop                             #  154   0x11d792  1      OPC=nop             
  nop                             #  155   0x11d793  1      OPC=nop             
.L_11d780:                        #        0x11d794  0      OPC=<label>         
  subl $0x1, %r12d                #  156   0x11d794  4      OPC=subl_r32_imm8   
  jns .L_11d700                   #  157   0x11d798  6      OPC=jns_label_1     
  nop                             #  158   0x11d79e  1      OPC=nop             
  nop                             #  159   0x11d79f  1      OPC=nop             
  nop                             #  160   0x11d7a0  1      OPC=nop             
  nop                             #  161   0x11d7a1  1      OPC=nop             
  nop                             #  162   0x11d7a2  1      OPC=nop             
  nop                             #  163   0x11d7a3  1      OPC=nop             
  nop                             #  164   0x11d7a4  1      OPC=nop             
  nop                             #  165   0x11d7a5  1      OPC=nop             
  nop                             #  166   0x11d7a6  1      OPC=nop             
  nop                             #  167   0x11d7a7  1      OPC=nop             
  nop                             #  168   0x11d7a8  1      OPC=nop             
  nop                             #  169   0x11d7a9  1      OPC=nop             
  nop                             #  170   0x11d7aa  1      OPC=nop             
  nop                             #  171   0x11d7ab  1      OPC=nop             
  nop                             #  172   0x11d7ac  1      OPC=nop             
  nop                             #  173   0x11d7ad  1      OPC=nop             
  nop                             #  174   0x11d7ae  1      OPC=nop             
  nop                             #  175   0x11d7af  1      OPC=nop             
  nop                             #  176   0x11d7b0  1      OPC=nop             
  nop                             #  177   0x11d7b1  1      OPC=nop             
  nop                             #  178   0x11d7b2  1      OPC=nop             
  nop                             #  179   0x11d7b3  1      OPC=nop             
.L_11d7a0:                        #        0x11d7b4  0      OPC=<label>         
  movl %r13d, %r13d               #  180   0x11d7b4  3      OPC=movl_r32_r32    
  movl (%r15,%r13,1), %r13d       #  181   0x11d7b7  4      OPC=movl_r32_m32    
  testq %r13, %r13                #  182   0x11d7bb  3      OPC=testq_r64_r64   
  jne .L_11d6e0                   #  183   0x11d7be  6      OPC=jne_label_1     
  nop                             #  184   0x11d7c4  1      OPC=nop             
  nop                             #  185   0x11d7c5  1      OPC=nop             
  nop                             #  186   0x11d7c6  1      OPC=nop             
  nop                             #  187   0x11d7c7  1      OPC=nop             
  nop                             #  188   0x11d7c8  1      OPC=nop             
  nop                             #  189   0x11d7c9  1      OPC=nop             
  nop                             #  190   0x11d7ca  1      OPC=nop             
  nop                             #  191   0x11d7cb  1      OPC=nop             
  nop                             #  192   0x11d7cc  1      OPC=nop             
  nop                             #  193   0x11d7cd  1      OPC=nop             
  nop                             #  194   0x11d7ce  1      OPC=nop             
  nop                             #  195   0x11d7cf  1      OPC=nop             
  nop                             #  196   0x11d7d0  1      OPC=nop             
  nop                             #  197   0x11d7d1  1      OPC=nop             
  nop                             #  198   0x11d7d2  1      OPC=nop             
  nop                             #  199   0x11d7d3  1      OPC=nop             
.L_11d7c0:                        #        0x11d7d4  0      OPC=<label>         
  nop                             #  200   0x11d7d4  1      OPC=nop             
  nop                             #  201   0x11d7d5  1      OPC=nop             
  nop                             #  202   0x11d7d6  1      OPC=nop             
  nop                             #  203   0x11d7d7  1      OPC=nop             
  nop                             #  204   0x11d7d8  1      OPC=nop             
  nop                             #  205   0x11d7d9  1      OPC=nop             
  nop                             #  206   0x11d7da  1      OPC=nop             
  nop                             #  207   0x11d7db  1      OPC=nop             
  nop                             #  208   0x11d7dc  1      OPC=nop             
  nop                             #  209   0x11d7dd  1      OPC=nop             
  nop                             #  210   0x11d7de  1      OPC=nop             
  nop                             #  211   0x11d7df  1      OPC=nop             
  nop                             #  212   0x11d7e0  1      OPC=nop             
  nop                             #  213   0x11d7e1  1      OPC=nop             
  nop                             #  214   0x11d7e2  1      OPC=nop             
  nop                             #  215   0x11d7e3  1      OPC=nop             
  nop                             #  216   0x11d7e4  1      OPC=nop             
  nop                             #  217   0x11d7e5  1      OPC=nop             
  nop                             #  218   0x11d7e6  1      OPC=nop             
  nop                             #  219   0x11d7e7  1      OPC=nop             
  nop                             #  220   0x11d7e8  1      OPC=nop             
  nop                             #  221   0x11d7e9  1      OPC=nop             
  nop                             #  222   0x11d7ea  1      OPC=nop             
  nop                             #  223   0x11d7eb  1      OPC=nop             
  nop                             #  224   0x11d7ec  1      OPC=nop             
  nop                             #  225   0x11d7ed  1      OPC=nop             
  nop                             #  226   0x11d7ee  1      OPC=nop             
  callq .__sfp_lock_release       #  227   0x11d7ef  5      OPC=callq_label     
  addl $0x18, %esp                #  228   0x11d7f4  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                 #  229   0x11d7f7  3      OPC=addq_r64_r64    
  movl %r14d, %eax                #  230   0x11d7fa  3      OPC=movl_r32_r32    
  popq %rbx                       #  231   0x11d7fd  1      OPC=popq_r64_1      
  popq %r12                       #  232   0x11d7fe  2      OPC=popq_r64_1      
  popq %r13                       #  233   0x11d800  2      OPC=popq_r64_1      
  popq %r14                       #  234   0x11d802  2      OPC=popq_r64_1      
  popq %r11                       #  235   0x11d804  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d         #  236   0x11d806  7      OPC=andl_r32_imm32  
  nop                             #  237   0x11d80d  1      OPC=nop             
  nop                             #  238   0x11d80e  1      OPC=nop             
  nop                             #  239   0x11d80f  1      OPC=nop             
  nop                             #  240   0x11d810  1      OPC=nop             
  addq %r15, %r11                 #  241   0x11d811  3      OPC=addq_r64_r64    
  jmpq %r11                       #  242   0x11d814  3      OPC=jmpq_r64        
  nop                             #  243   0x11d817  1      OPC=nop             
  nop                             #  244   0x11d818  1      OPC=nop             
  nop                             #  245   0x11d819  1      OPC=nop             
  nop                             #  246   0x11d81a  1      OPC=nop             
.L_11d800:                        #        0x11d81b  0      OPC=<label>         
  movl %eax, %edi                 #  247   0x11d81b  2      OPC=movl_r32_r32    
  nop                             #  248   0x11d81d  1      OPC=nop             
  nop                             #  249   0x11d81e  1      OPC=nop             
  nop                             #  250   0x11d81f  1      OPC=nop             
  nop                             #  251   0x11d820  1      OPC=nop             
  nop                             #  252   0x11d821  1      OPC=nop             
  nop                             #  253   0x11d822  1      OPC=nop             
  nop                             #  254   0x11d823  1      OPC=nop             
  nop                             #  255   0x11d824  1      OPC=nop             
  nop                             #  256   0x11d825  1      OPC=nop             
  nop                             #  257   0x11d826  1      OPC=nop             
  nop                             #  258   0x11d827  1      OPC=nop             
  nop                             #  259   0x11d828  1      OPC=nop             
  nop                             #  260   0x11d829  1      OPC=nop             
  nop                             #  261   0x11d82a  1      OPC=nop             
  nop                             #  262   0x11d82b  1      OPC=nop             
  nop                             #  263   0x11d82c  1      OPC=nop             
  nop                             #  264   0x11d82d  1      OPC=nop             
  nop                             #  265   0x11d82e  1      OPC=nop             
  nop                             #  266   0x11d82f  1      OPC=nop             
  nop                             #  267   0x11d830  1      OPC=nop             
  nop                             #  268   0x11d831  1      OPC=nop             
  nop                             #  269   0x11d832  1      OPC=nop             
  nop                             #  270   0x11d833  1      OPC=nop             
  nop                             #  271   0x11d834  1      OPC=nop             
  nop                             #  272   0x11d835  1      OPC=nop             
  callq .__sinit                  #  273   0x11d836  5      OPC=callq_label     
  jmpq .L_11d6c0                  #  274   0x11d83b  5      OPC=jmpq_label_1    
  nop                             #  275   0x11d840  1      OPC=nop             
  nop                             #  276   0x11d841  1      OPC=nop             
  nop                             #  277   0x11d842  1      OPC=nop             
  nop                             #  278   0x11d843  1      OPC=nop             
  nop                             #  279   0x11d844  1      OPC=nop             
  nop                             #  280   0x11d845  1      OPC=nop             
  nop                             #  281   0x11d846  1      OPC=nop             
  nop                             #  282   0x11d847  1      OPC=nop             
  nop                             #  283   0x11d848  1      OPC=nop             
  nop                             #  284   0x11d849  1      OPC=nop             
  nop                             #  285   0x11d84a  1      OPC=nop             
  nop                             #  286   0x11d84b  1      OPC=nop             
  nop                             #  287   0x11d84c  1      OPC=nop             
  nop                             #  288   0x11d84d  1      OPC=nop             
  nop                             #  289   0x11d84e  1      OPC=nop             
  nop                             #  290   0x11d84f  1      OPC=nop             
  nop                             #  291   0x11d850  1      OPC=nop             
  nop                             #  292   0x11d851  1      OPC=nop             
  nop                             #  293   0x11d852  1      OPC=nop             
  nop                             #  294   0x11d853  1      OPC=nop             
  nop                             #  295   0x11d854  1      OPC=nop             
  nop                             #  296   0x11d855  1      OPC=nop             
  nop                             #  297   0x11d856  1      OPC=nop             
  nop                             #  298   0x11d857  1      OPC=nop             
  nop                             #  299   0x11d858  1      OPC=nop             
  nop                             #  300   0x11d859  1      OPC=nop             
  nop                             #  301   0x11d85a  1      OPC=nop             
                                                                                
.size _fwalk_reent, .-_fwalk_reent
