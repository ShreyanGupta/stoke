  .text
  .globl mallopt
  .type mallopt, @function

#! file-offset 0x1515c0
#! rip-offset  0x1115c0
#! capacity    320 bytes

# Text                        #  Line  RIP       Bytes  Opcode              
.mallopt:                     #        0x1115c0  0      OPC=<label>         
  subl $0x18, %esp            #  1     0x1115c0  3      OPC=subl_r32_imm8   
  addq %r15, %rsp             #  2     0x1115c3  3      OPC=addq_r64_r64    
  movl 0xff67584(%rip), %edx  #  3     0x1115c6  6      OPC=movl_r32_m32    
  testl %edx, %edx            #  4     0x1115cc  2      OPC=testl_r32_r32   
  je .L_111660                #  5     0x1115ce  6      OPC=je_label_1      
  cmpl $0xfffffffe, %edi      #  6     0x1115d4  6      OPC=cmpl_r32_imm32  
  nop                         #  7     0x1115da  1      OPC=nop             
  nop                         #  8     0x1115db  1      OPC=nop             
  nop                         #  9     0x1115dc  1      OPC=nop             
  je .L_1116a0                #  10    0x1115dd  6      OPC=je_label_1      
  nop                         #  11    0x1115e3  1      OPC=nop             
  nop                         #  12    0x1115e4  1      OPC=nop             
  nop                         #  13    0x1115e5  1      OPC=nop             
.L_1115e0:                    #        0x1115e6  0      OPC=<label>         
  cmpl $0xffffffff, %edi      #  14    0x1115e6  6      OPC=cmpl_r32_imm32  
  nop                         #  15    0x1115ec  1      OPC=nop             
  nop                         #  16    0x1115ed  1      OPC=nop             
  nop                         #  17    0x1115ee  1      OPC=nop             
  je .L_1116e0                #  18    0x1115ef  6      OPC=je_label_1      
  cmpl $0xfffffffd, %edi      #  19    0x1115f5  6      OPC=cmpl_r32_imm32  
  nop                         #  20    0x1115fb  1      OPC=nop             
  nop                         #  21    0x1115fc  1      OPC=nop             
  nop                         #  22    0x1115fd  1      OPC=nop             
  je .L_111640                #  23    0x1115fe  2      OPC=je_label        
  nop                         #  24    0x111600  1      OPC=nop             
  nop                         #  25    0x111601  1      OPC=nop             
  nop                         #  26    0x111602  1      OPC=nop             
  nop                         #  27    0x111603  1      OPC=nop             
  nop                         #  28    0x111604  1      OPC=nop             
  nop                         #  29    0x111605  1      OPC=nop             
  nop                         #  30    0x111606  1      OPC=nop             
  nop                         #  31    0x111607  1      OPC=nop             
  nop                         #  32    0x111608  1      OPC=nop             
  nop                         #  33    0x111609  1      OPC=nop             
  nop                         #  34    0x11160a  1      OPC=nop             
  nop                         #  35    0x11160b  1      OPC=nop             
  nop                         #  36    0x11160c  1      OPC=nop             
  nop                         #  37    0x11160d  1      OPC=nop             
  nop                         #  38    0x11160e  1      OPC=nop             
  nop                         #  39    0x11160f  1      OPC=nop             
  nop                         #  40    0x111610  1      OPC=nop             
  nop                         #  41    0x111611  1      OPC=nop             
.L_111600:                    #        0x111612  0      OPC=<label>         
  xorl %eax, %eax             #  42    0x111612  2      OPC=xorl_r32_r32    
  nop                         #  43    0x111614  1      OPC=nop             
  nop                         #  44    0x111615  1      OPC=nop             
  nop                         #  45    0x111616  1      OPC=nop             
  nop                         #  46    0x111617  1      OPC=nop             
  nop                         #  47    0x111618  1      OPC=nop             
  nop                         #  48    0x111619  1      OPC=nop             
  nop                         #  49    0x11161a  1      OPC=nop             
  nop                         #  50    0x11161b  1      OPC=nop             
  nop                         #  51    0x11161c  1      OPC=nop             
  nop                         #  52    0x11161d  1      OPC=nop             
  nop                         #  53    0x11161e  1      OPC=nop             
  nop                         #  54    0x11161f  1      OPC=nop             
  nop                         #  55    0x111620  1      OPC=nop             
  nop                         #  56    0x111621  1      OPC=nop             
  nop                         #  57    0x111622  1      OPC=nop             
  nop                         #  58    0x111623  1      OPC=nop             
  nop                         #  59    0x111624  1      OPC=nop             
  nop                         #  60    0x111625  1      OPC=nop             
  nop                         #  61    0x111626  1      OPC=nop             
  nop                         #  62    0x111627  1      OPC=nop             
  nop                         #  63    0x111628  1      OPC=nop             
  nop                         #  64    0x111629  1      OPC=nop             
  nop                         #  65    0x11162a  1      OPC=nop             
  nop                         #  66    0x11162b  1      OPC=nop             
  nop                         #  67    0x11162c  1      OPC=nop             
  nop                         #  68    0x11162d  1      OPC=nop             
  nop                         #  69    0x11162e  1      OPC=nop             
  nop                         #  70    0x11162f  1      OPC=nop             
  nop                         #  71    0x111630  1      OPC=nop             
  nop                         #  72    0x111631  1      OPC=nop             
.L_111620:                    #        0x111632  0      OPC=<label>         
  addl $0x18, %esp            #  73    0x111632  3      OPC=addl_r32_imm8   
  addq %r15, %rsp             #  74    0x111635  3      OPC=addq_r64_r64    
  popq %r11                   #  75    0x111638  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d     #  76    0x11163a  7      OPC=andl_r32_imm32  
  nop                         #  77    0x111641  1      OPC=nop             
  nop                         #  78    0x111642  1      OPC=nop             
  nop                         #  79    0x111643  1      OPC=nop             
  nop                         #  80    0x111644  1      OPC=nop             
  addq %r15, %r11             #  81    0x111645  3      OPC=addq_r64_r64    
  jmpq %r11                   #  82    0x111648  3      OPC=jmpq_r64        
  nop                         #  83    0x11164b  1      OPC=nop             
  nop                         #  84    0x11164c  1      OPC=nop             
  nop                         #  85    0x11164d  1      OPC=nop             
  nop                         #  86    0x11164e  1      OPC=nop             
  nop                         #  87    0x11164f  1      OPC=nop             
  nop                         #  88    0x111650  1      OPC=nop             
  nop                         #  89    0x111651  1      OPC=nop             
  nop                         #  90    0x111652  1      OPC=nop             
  nop                         #  91    0x111653  1      OPC=nop             
  nop                         #  92    0x111654  1      OPC=nop             
  nop                         #  93    0x111655  1      OPC=nop             
  nop                         #  94    0x111656  1      OPC=nop             
  nop                         #  95    0x111657  1      OPC=nop             
  nop                         #  96    0x111658  1      OPC=nop             
.L_111640:                    #        0x111659  0      OPC=<label>         
  movl %esi, 0xff67516(%rip)  #  97    0x111659  6      OPC=movl_m32_r32    
  addl $0x18, %esp            #  98    0x11165f  3      OPC=addl_r32_imm8   
  addq %r15, %rsp             #  99    0x111662  3      OPC=addq_r64_r64    
  movl $0x1, %eax             #  100   0x111665  5      OPC=movl_r32_imm32  
  popq %r11                   #  101   0x11166a  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d     #  102   0x11166c  7      OPC=andl_r32_imm32  
  nop                         #  103   0x111673  1      OPC=nop             
  nop                         #  104   0x111674  1      OPC=nop             
  nop                         #  105   0x111675  1      OPC=nop             
  nop                         #  106   0x111676  1      OPC=nop             
  addq %r15, %r11             #  107   0x111677  3      OPC=addq_r64_r64    
  jmpq %r11                   #  108   0x11167a  3      OPC=jmpq_r64        
  nop                         #  109   0x11167d  1      OPC=nop             
  nop                         #  110   0x11167e  1      OPC=nop             
  nop                         #  111   0x11167f  1      OPC=nop             
.L_111660:                    #        0x111680  0      OPC=<label>         
  movl %esi, (%rsp)           #  112   0x111680  3      OPC=movl_m32_r32    
  movl %edi, 0x8(%rsp)        #  113   0x111683  4      OPC=movl_m32_r32    
  nop                         #  114   0x111687  1      OPC=nop             
  nop                         #  115   0x111688  1      OPC=nop             
  nop                         #  116   0x111689  1      OPC=nop             
  nop                         #  117   0x11168a  1      OPC=nop             
  nop                         #  118   0x11168b  1      OPC=nop             
  nop                         #  119   0x11168c  1      OPC=nop             
  nop                         #  120   0x11168d  1      OPC=nop             
  nop                         #  121   0x11168e  1      OPC=nop             
  nop                         #  122   0x11168f  1      OPC=nop             
  nop                         #  123   0x111690  1      OPC=nop             
  nop                         #  124   0x111691  1      OPC=nop             
  nop                         #  125   0x111692  1      OPC=nop             
  nop                         #  126   0x111693  1      OPC=nop             
  nop                         #  127   0x111694  1      OPC=nop             
  nop                         #  128   0x111695  1      OPC=nop             
  nop                         #  129   0x111696  1      OPC=nop             
  nop                         #  130   0x111697  1      OPC=nop             
  nop                         #  131   0x111698  1      OPC=nop             
  nop                         #  132   0x111699  1      OPC=nop             
  nop                         #  133   0x11169a  1      OPC=nop             
  callq .init_mparams         #  134   0x11169b  5      OPC=callq_label     
  movl 0x8(%rsp), %edi        #  135   0x1116a0  4      OPC=movl_r32_m32    
  movl (%rsp), %esi           #  136   0x1116a4  3      OPC=movl_r32_m32    
  cmpl $0xfffffffe, %edi      #  137   0x1116a7  6      OPC=cmpl_r32_imm32  
  nop                         #  138   0x1116ad  1      OPC=nop             
  nop                         #  139   0x1116ae  1      OPC=nop             
  nop                         #  140   0x1116af  1      OPC=nop             
  jne .L_1115e0               #  141   0x1116b0  6      OPC=jne_label_1     
  nop                         #  142   0x1116b6  1      OPC=nop             
  nop                         #  143   0x1116b7  1      OPC=nop             
  nop                         #  144   0x1116b8  1      OPC=nop             
  nop                         #  145   0x1116b9  1      OPC=nop             
  nop                         #  146   0x1116ba  1      OPC=nop             
  nop                         #  147   0x1116bb  1      OPC=nop             
  nop                         #  148   0x1116bc  1      OPC=nop             
  nop                         #  149   0x1116bd  1      OPC=nop             
  nop                         #  150   0x1116be  1      OPC=nop             
  nop                         #  151   0x1116bf  1      OPC=nop             
  nop                         #  152   0x1116c0  1      OPC=nop             
  nop                         #  153   0x1116c1  1      OPC=nop             
  nop                         #  154   0x1116c2  1      OPC=nop             
  nop                         #  155   0x1116c3  1      OPC=nop             
  nop                         #  156   0x1116c4  1      OPC=nop             
  nop                         #  157   0x1116c5  1      OPC=nop             
.L_1116a0:                    #        0x1116c6  0      OPC=<label>         
  cmpl 0xff674ae(%rip), %esi  #  158   0x1116c6  6      OPC=cmpl_r32_m32    
  jb .L_111600                #  159   0x1116cc  6      OPC=jb_label_1      
  leal -0x1(%rsi), %eax       #  160   0x1116d2  3      OPC=leal_r32_m16    
  testl %esi, %eax            #  161   0x1116d5  2      OPC=testl_r32_r32   
  jne .L_111600               #  162   0x1116d7  6      OPC=jne_label_1     
  movl %esi, 0xff6749b(%rip)  #  163   0x1116dd  6      OPC=movl_m32_r32    
  nop                         #  164   0x1116e3  1      OPC=nop             
  nop                         #  165   0x1116e4  1      OPC=nop             
  nop                         #  166   0x1116e5  1      OPC=nop             
  movl $0x1, %eax             #  167   0x1116e6  5      OPC=movl_r32_imm32  
  jmpq .L_111620              #  168   0x1116eb  5      OPC=jmpq_label_1    
  nop                         #  169   0x1116f0  1      OPC=nop             
  nop                         #  170   0x1116f1  1      OPC=nop             
  nop                         #  171   0x1116f2  1      OPC=nop             
  nop                         #  172   0x1116f3  1      OPC=nop             
  nop                         #  173   0x1116f4  1      OPC=nop             
  nop                         #  174   0x1116f5  1      OPC=nop             
  nop                         #  175   0x1116f6  1      OPC=nop             
  nop                         #  176   0x1116f7  1      OPC=nop             
  nop                         #  177   0x1116f8  1      OPC=nop             
  nop                         #  178   0x1116f9  1      OPC=nop             
  nop                         #  179   0x1116fa  1      OPC=nop             
  nop                         #  180   0x1116fb  1      OPC=nop             
  nop                         #  181   0x1116fc  1      OPC=nop             
  nop                         #  182   0x1116fd  1      OPC=nop             
  nop                         #  183   0x1116fe  1      OPC=nop             
  nop                         #  184   0x1116ff  1      OPC=nop             
  nop                         #  185   0x111700  1      OPC=nop             
  nop                         #  186   0x111701  1      OPC=nop             
  nop                         #  187   0x111702  1      OPC=nop             
  nop                         #  188   0x111703  1      OPC=nop             
  nop                         #  189   0x111704  1      OPC=nop             
  nop                         #  190   0x111705  1      OPC=nop             
.L_1116e0:                    #        0x111706  0      OPC=<label>         
  movl %esi, 0xff6747a(%rip)  #  191   0x111706  6      OPC=movl_m32_r32    
  addl $0x18, %esp            #  192   0x11170c  3      OPC=addl_r32_imm8   
  addq %r15, %rsp             #  193   0x11170f  3      OPC=addq_r64_r64    
  movl $0x1, %eax             #  194   0x111712  5      OPC=movl_r32_imm32  
  popq %r11                   #  195   0x111717  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d     #  196   0x111719  7      OPC=andl_r32_imm32  
  nop                         #  197   0x111720  1      OPC=nop             
  nop                         #  198   0x111721  1      OPC=nop             
  nop                         #  199   0x111722  1      OPC=nop             
  nop                         #  200   0x111723  1      OPC=nop             
  addq %r15, %r11             #  201   0x111724  3      OPC=addq_r64_r64    
  jmpq %r11                   #  202   0x111727  3      OPC=jmpq_r64        
  nop                         #  203   0x11172a  1      OPC=nop             
  nop                         #  204   0x11172b  1      OPC=nop             
  nop                         #  205   0x11172c  1      OPC=nop             
                                                                            
.size mallopt, .-mallopt

