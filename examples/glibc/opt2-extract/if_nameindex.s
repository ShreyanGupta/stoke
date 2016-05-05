  .text
  .globl if_nameindex
  .type if_nameindex, @function

#! file-offset 0xff300
#! rip-offset  0xff300
#! capacity    720 bytes

# Text                          #  Line  RIP      Bytes  Opcode              
.if_nameindex:                  #        0xff300  0      OPC=<label>         
  pushq %r15                    #  1     0xff300  2      OPC=pushq_r64_1     
  pushq %r14                    #  2     0xff302  2      OPC=pushq_r64_1     
  pushq %r13                    #  3     0xff304  2      OPC=pushq_r64_1     
  pushq %r12                    #  4     0xff306  2      OPC=pushq_r64_1     
  pushq %rbp                    #  5     0xff308  1      OPC=pushq_r64_1     
  pushq %rbx                    #  6     0xff309  1      OPC=pushq_r64_1     
  subq $0x38, %rsp              #  7     0xff30a  4      OPC=subq_r64_imm8   
  leaq 0x10(%rsp), %rbx         #  8     0xff30e  5      OPC=leaq_r64_m16    
  movl $0x0, 0x10(%rsp)         #  9     0xff313  8      OPC=movl_m32_imm32  
  movl $0x0, 0x14(%rsp)         #  10    0xff31b  8      OPC=movl_m32_imm32  
  movl $0x0, 0x18(%rsp)         #  11    0xff323  8      OPC=movl_m32_imm32  
  movq $0x0, 0x20(%rsp)         #  12    0xff32b  9      OPC=movq_m64_imm32  
  movq %rbx, %rdi               #  13    0xff334  3      OPC=movq_r64_r64    
  movq $0x0, 0x28(%rsp)         #  14    0xff337  9      OPC=movq_m64_imm32  
  callq .__netlink_open         #  15    0xff340  5      OPC=callq_label     
  testl %eax, %eax              #  16    0xff345  2      OPC=testl_r32_r32   
  js .L_ff598                   #  17    0xff347  6      OPC=js_label_1      
  movl $0x12, %esi              #  18    0xff34d  5      OPC=movl_r32_imm32  
  movq %rbx, %rdi               #  19    0xff352  3      OPC=movq_r64_r64    
  xorl %r15d, %r15d             #  20    0xff355  3      OPC=xorl_r32_r32    
  callq .__netlink_request      #  21    0xff358  5      OPC=callq_label     
  testl %eax, %eax              #  22    0xff35d  2      OPC=testl_r32_r32   
  js .L_ff535                   #  23    0xff35f  6      OPC=js_label_1      
  movq 0x20(%rsp), %r12         #  24    0xff365  5      OPC=movq_r64_m64    
  testq %r12, %r12              #  25    0xff36a  3      OPC=testq_r64_r64   
  je .L_ff5a9                   #  26    0xff36d  6      OPC=je_label_1      
  movl 0x14(%rsp), %edi         #  27    0xff373  4      OPC=movl_r32_m32    
  movq %r12, %r8                #  28    0xff377  3      OPC=movq_r64_r64    
  xorl %r9d, %r9d               #  29    0xff37a  3      OPC=xorl_r32_r32    
  nop                           #  30    0xff37d  1      OPC=nop             
  nop                           #  31    0xff37e  1      OPC=nop             
  nop                           #  32    0xff37f  1      OPC=nop             
.L_ff380:                       #        0xff380  0      OPC=<label>         
  movq 0x8(%r8), %rdx           #  33    0xff380  4      OPC=movq_r64_m64    
  movq 0x10(%r8), %rcx          #  34    0xff384  4      OPC=movq_r64_m64    
  testq %rdx, %rdx              #  35    0xff388  3      OPC=testq_r64_r64   
  je .L_ff3e8                   #  36    0xff38b  2      OPC=je_label        
  cmpq $0xf, %rcx               #  37    0xff38d  4      OPC=cmpq_r64_imm8   
  ja .L_ff3aa                   #  38    0xff391  2      OPC=ja_label        
  jmpq .L_ff3e8                 #  39    0xff393  2      OPC=jmpq_label      
  nop                           #  40    0xff395  1      OPC=nop             
  nop                           #  41    0xff396  1      OPC=nop             
  nop                           #  42    0xff397  1      OPC=nop             
.L_ff398:                       #        0xff398  0      OPC=<label>         
  addl $0x3, %eax               #  43    0xff398  3      OPC=addl_r32_imm8   
  andl $0xfffffffc, %eax        #  44    0xff39b  6      OPC=andl_r32_imm32  
  nop                           #  45    0xff3a1  1      OPC=nop             
  nop                           #  46    0xff3a2  1      OPC=nop             
  nop                           #  47    0xff3a3  1      OPC=nop             
  subq %rax, %rcx               #  48    0xff3a4  3      OPC=subq_r64_r64    
  addq %rax, %rdx               #  49    0xff3a7  3      OPC=addq_r64_r64    
  cmpq $0xf, %rcx               #  50    0xff3aa  4      OPC=cmpq_r64_imm8   
  jbe .L_ff3e8                  #  51    0xff3ae  2      OPC=jbe_label       
.L_ff3aa:                       #        0xff3b0  0      OPC=<label>         
  movl (%rdx), %eax             #  52    0xff3b0  2      OPC=movl_r32_m32    
  cmpl $0xf, %eax               #  53    0xff3b2  3      OPC=cmpl_r32_imm8   
  jbe .L_ff3e8                  #  54    0xff3b5  2      OPC=jbe_label       
  movl %eax, %esi               #  55    0xff3b7  2      OPC=movl_r32_r32    
  cmpq %rsi, %rcx               #  56    0xff3b9  3      OPC=cmpq_r64_r64    
  jb .L_ff3e8                   #  57    0xff3bc  2      OPC=jb_label        
  cmpl %edi, 0xc(%rdx)          #  58    0xff3be  3      OPC=cmpl_m32_r32    
  jne .L_ff398                  #  59    0xff3c1  2      OPC=jne_label       
  movl 0x18(%r8), %esi          #  60    0xff3c3  4      OPC=movl_r32_m32    
  cmpl %esi, 0x8(%rdx)          #  61    0xff3c7  3      OPC=cmpl_m32_r32    
  jne .L_ff398                  #  62    0xff3ca  2      OPC=jne_label       
  movzwl 0x4(%rdx), %esi        #  63    0xff3cc  4      OPC=movzwl_r32_m16  
  cmpw $0x3, %si                #  64    0xff3d0  4      OPC=cmpw_r16_imm8   
  je .L_ff3e8                   #  65    0xff3d4  2      OPC=je_label        
  cmpw $0x10, %si               #  66    0xff3d6  4      OPC=cmpw_r16_imm8   
  sete %sil                     #  67    0xff3da  4      OPC=sete_r8         
  movzbl %sil, %esi             #  68    0xff3de  4      OPC=movzbl_r32_r8   
  addl %esi, %r9d               #  69    0xff3e2  3      OPC=addl_r32_r32    
  jmpq .L_ff398                 #  70    0xff3e5  2      OPC=jmpq_label      
  nop                           #  71    0xff3e7  1      OPC=nop             
  nop                           #  72    0xff3e8  1      OPC=nop             
  nop                           #  73    0xff3e9  1      OPC=nop             
  nop                           #  74    0xff3ea  1      OPC=nop             
  nop                           #  75    0xff3eb  1      OPC=nop             
  nop                           #  76    0xff3ec  1      OPC=nop             
  nop                           #  77    0xff3ed  1      OPC=nop             
.L_ff3e8:                       #        0xff3ee  0      OPC=<label>         
  movq (%r8), %r8               #  78    0xff3ee  3      OPC=movq_r64_m64    
  testq %r8, %r8                #  79    0xff3f1  3      OPC=testq_r64_r64   
  jne .L_ff380                  #  80    0xff3f4  2      OPC=jne_label       
  leal 0x1(%r9), %edi           #  81    0xff3f6  4      OPC=leal_r32_m16    
  xorl %r14d, %r14d             #  82    0xff3fa  3      OPC=xorl_r32_r32    
  shlq $0x4, %rdi               #  83    0xff3fd  4      OPC=shlq_r64_imm8   
  callq .memalign_plt           #  84    0xff401  5      OPC=callq_label     
  testq %rax, %rax              #  85    0xff406  3      OPC=testq_r64_r64   
  movq %rax, %r15               #  86    0xff409  3      OPC=movq_r64_r64    
  je .L_ff585                   #  87    0xff40c  6      OPC=je_label_1      
  nop                           #  88    0xff412  1      OPC=nop             
  nop                           #  89    0xff413  1      OPC=nop             
  nop                           #  90    0xff414  1      OPC=nop             
  nop                           #  91    0xff415  1      OPC=nop             
.L_ff410:                       #        0xff416  0      OPC=<label>         
  movq 0x8(%r12), %rbp          #  92    0xff416  5      OPC=movq_r64_m64    
  movq 0x10(%r12), %r13         #  93    0xff41b  5      OPC=movq_r64_m64    
  testq %rbp, %rbp              #  94    0xff420  3      OPC=testq_r64_r64   
  je .L_ff510                   #  95    0xff423  6      OPC=je_label_1      
  cmpq $0xf, %r13               #  96    0xff429  4      OPC=cmpq_r64_imm8   
  jbe .L_ff510                  #  97    0xff42d  6      OPC=jbe_label_1     
  movl (%rbp), %edx             #  98    0xff433  3      OPC=movl_r32_m32    
  cmpl $0xf, %edx               #  99    0xff436  3      OPC=cmpl_r32_imm8   
  jbe .L_ff510                  #  100   0xff439  6      OPC=jbe_label_1     
  cmpq %rdx, %r13               #  101   0xff43f  3      OPC=cmpq_r64_r64    
  jae .L_ff476                  #  102   0xff442  2      OPC=jae_label       
  jmpq .L_ff510                 #  103   0xff444  5      OPC=jmpq_label_1    
  nop                           #  104   0xff449  1      OPC=nop             
  nop                           #  105   0xff44a  1      OPC=nop             
  nop                           #  106   0xff44b  1      OPC=nop             
  nop                           #  107   0xff44c  1      OPC=nop             
  nop                           #  108   0xff44d  1      OPC=nop             
.L_ff448:                       #        0xff44e  0      OPC=<label>         
  movl (%rbp), %eax             #  109   0xff44e  3      OPC=movl_r32_m32    
  leal 0x3(%rax), %edx          #  110   0xff451  3      OPC=leal_r32_m16    
  andl $0xfffffffc, %edx        #  111   0xff454  6      OPC=andl_r32_imm32  
  nop                           #  112   0xff45a  1      OPC=nop             
  nop                           #  113   0xff45b  1      OPC=nop             
  nop                           #  114   0xff45c  1      OPC=nop             
  subq %rdx, %r13               #  115   0xff45d  3      OPC=subq_r64_r64    
  addq %rdx, %rbp               #  116   0xff460  3      OPC=addq_r64_r64    
  cmpq $0xf, %r13               #  117   0xff463  4      OPC=cmpq_r64_imm8   
  jbe .L_ff510                  #  118   0xff467  6      OPC=jbe_label_1     
  movl (%rbp), %edx             #  119   0xff46d  3      OPC=movl_r32_m32    
  cmpl $0xf, %edx               #  120   0xff470  3      OPC=cmpl_r32_imm8   
  jbe .L_ff510                  #  121   0xff473  6      OPC=jbe_label_1     
  cmpq %r13, %rdx               #  122   0xff479  3      OPC=cmpq_r64_r64    
  ja .L_ff510                   #  123   0xff47c  6      OPC=ja_label_1      
.L_ff476:                       #        0xff482  0      OPC=<label>         
  movl 0x14(%rsp), %eax         #  124   0xff482  4      OPC=movl_r32_m32    
  cmpl %eax, 0xc(%rbp)          #  125   0xff486  3      OPC=cmpl_m32_r32    
  jne .L_ff448                  #  126   0xff489  2      OPC=jne_label       
  movl 0x18(%r12), %eax         #  127   0xff48b  5      OPC=movl_r32_m32    
  cmpl %eax, 0x8(%rbp)          #  128   0xff490  3      OPC=cmpl_m32_r32    
  jne .L_ff448                  #  129   0xff493  2      OPC=jne_label       
  movzwl 0x4(%rbp), %eax        #  130   0xff495  4      OPC=movzwl_r32_m16  
  cmpw $0x3, %ax                #  131   0xff499  4      OPC=cmpw_ax_imm16   
  je .L_ff510                   #  132   0xff49d  2      OPC=je_label        
  cmpw $0x10, %ax               #  133   0xff49f  4      OPC=cmpw_ax_imm16   
  jne .L_ff448                  #  134   0xff4a3  2      OPC=jne_label       
  movl %r14d, %ecx              #  135   0xff4a5  3      OPC=movl_r32_r32    
  movl 0x14(%rbp), %eax         #  136   0xff4a8  3      OPC=movl_r32_m32    
  subq $0x20, %rdx              #  137   0xff4ab  4      OPC=subq_r64_imm8   
  shlq $0x4, %rcx               #  138   0xff4af  4      OPC=shlq_r64_imm8   
  leaq 0x20(%rbp), %rsi         #  139   0xff4b3  4      OPC=leaq_r64_m16    
  addq %r15, %rcx               #  140   0xff4b7  3      OPC=addq_r64_r64    
  cmpq $0x3, %rdx               #  141   0xff4ba  4      OPC=cmpq_r64_imm8   
  movl %eax, (%rcx)             #  142   0xff4be  2      OPC=movl_m32_r32    
  jbe .L_ff505                  #  143   0xff4c0  2      OPC=jbe_label       
  movzwl 0x20(%rbp), %eax       #  144   0xff4c2  4      OPC=movzwl_r32_m16  
  cmpw $0x3, %ax                #  145   0xff4c6  4      OPC=cmpw_ax_imm16   
  jbe .L_ff505                  #  146   0xff4ca  2      OPC=jbe_label       
  movzwl %ax, %r8d              #  147   0xff4cc  4      OPC=movzwl_r32_r16  
  cmpq %r8, %rdx                #  148   0xff4d0  3      OPC=cmpq_r64_r64    
  jb .L_ff505                   #  149   0xff4d3  2      OPC=jb_label        
  cmpw $0x3, 0x22(%rbp)         #  150   0xff4d5  5      OPC=cmpw_m16_imm8   
  jne .L_ff4f1                  #  151   0xff4da  2      OPC=jne_label       
  jmpq .L_ff5c2                 #  152   0xff4dc  5      OPC=jmpq_label_1    
  nop                           #  153   0xff4e1  1      OPC=nop             
  nop                           #  154   0xff4e2  1      OPC=nop             
  nop                           #  155   0xff4e3  1      OPC=nop             
.L_ff4d8:                       #        0xff4e4  0      OPC=<label>         
  movzwl (%rsi), %eax           #  156   0xff4e4  3      OPC=movzwl_r32_m16  
  cmpw $0x3, %ax                #  157   0xff4e7  4      OPC=cmpw_ax_imm16   
  jbe .L_ff505                  #  158   0xff4eb  2      OPC=jbe_label       
  movzwl %ax, %r8d              #  159   0xff4ed  4      OPC=movzwl_r32_r16  
  cmpq %rdx, %r8                #  160   0xff4f1  3      OPC=cmpq_r64_r64    
  ja .L_ff505                   #  161   0xff4f4  2      OPC=ja_label        
  cmpw $0x3, 0x2(%rsi)          #  162   0xff4f6  5      OPC=cmpw_m16_imm8   
  je .L_ff557                   #  163   0xff4fb  2      OPC=je_label        
.L_ff4f1:                       #        0xff4fd  0      OPC=<label>         
  addl $0x3, %eax               #  164   0xff4fd  3      OPC=addl_r32_imm8   
  andl $0x1fffc, %eax           #  165   0xff500  5      OPC=andl_eax_imm32  
  subq %rax, %rdx               #  166   0xff505  3      OPC=subq_r64_r64    
  addq %rax, %rsi               #  167   0xff508  3      OPC=addq_r64_r64    
  cmpq $0x3, %rdx               #  168   0xff50b  4      OPC=cmpq_r64_imm8   
  ja .L_ff4d8                   #  169   0xff50f  2      OPC=ja_label        
.L_ff505:                       #        0xff511  0      OPC=<label>         
  addl $0x1, %r14d              #  170   0xff511  4      OPC=addl_r32_imm8   
  jmpq .L_ff448                 #  171   0xff515  5      OPC=jmpq_label_1    
  xchgw %ax, %ax                #  172   0xff51a  2      OPC=xchgw_ax_r16    
.L_ff510:                       #        0xff51c  0      OPC=<label>         
  movq (%r12), %r12             #  173   0xff51c  4      OPC=movq_r64_m64    
  testq %r12, %r12              #  174   0xff520  3      OPC=testq_r64_r64   
  jne .L_ff410                  #  175   0xff523  6      OPC=jne_label_1     
  movl %r14d, %eax              #  176   0xff529  3      OPC=movl_r32_r32    
  shlq $0x4, %rax               #  177   0xff52c  4      OPC=shlq_r64_imm8   
.L_ff524:                       #        0xff530  0      OPC=<label>         
  addq %r15, %rax               #  178   0xff530  3      OPC=addq_r64_r64    
  movl $0x0, (%rax)             #  179   0xff533  6      OPC=movl_m32_imm32  
  movq $0x0, 0x8(%rax)          #  180   0xff539  8      OPC=movq_m64_imm32  
.L_ff535:                       #        0xff541  0      OPC=<label>         
  movq %rbx, %rdi               #  181   0xff541  3      OPC=movq_r64_r64    
  callq .__netlink_free_handle  #  182   0xff544  5      OPC=callq_label     
  movq %rbx, %rdi               #  183   0xff549  3      OPC=movq_r64_r64    
  callq .__netlink_close        #  184   0xff54c  5      OPC=callq_label     
  addq $0x38, %rsp              #  185   0xff551  4      OPC=addq_r64_imm8   
  movq %r15, %rax               #  186   0xff555  3      OPC=movq_r64_r64    
  popq %rbx                     #  187   0xff558  1      OPC=popq_r64_1      
  popq %rbp                     #  188   0xff559  1      OPC=popq_r64_1      
  popq %r12                     #  189   0xff55a  2      OPC=popq_r64_1      
  popq %r13                     #  190   0xff55c  2      OPC=popq_r64_1      
  popq %r14                     #  191   0xff55e  2      OPC=popq_r64_1      
  popq %r15                     #  192   0xff560  2      OPC=popq_r64_1      
  retq                          #  193   0xff562  1      OPC=retq            
.L_ff557:                       #        0xff563  0      OPC=<label>         
  leaq 0x4(%rsi), %rdi          #  194   0xff563  4      OPC=leaq_r64_m16    
  leaq -0x4(%r8), %rsi          #  195   0xff567  4      OPC=leaq_r64_m16    
.L_ff55f:                       #        0xff56b  0      OPC=<label>         
  movq %rcx, 0x8(%rsp)          #  196   0xff56b  5      OPC=movq_m64_r64    
  callq .__strndup              #  197   0xff570  5      OPC=callq_label     
  movq 0x8(%rsp), %rcx          #  198   0xff575  5      OPC=movq_r64_m64    
  testq %rax, %rax              #  199   0xff57a  3      OPC=testq_r64_r64   
  movq %rax, 0x8(%rcx)          #  200   0xff57d  4      OPC=movq_m64_r64    
  jne .L_ff505                  #  201   0xff581  2      OPC=jne_label       
  movl $0x0, (%rcx)             #  202   0xff583  6      OPC=movl_m32_imm32  
  movq %r15, %rdi               #  203   0xff589  3      OPC=movq_r64_r64    
  callq .if_freenameindex       #  204   0xff58c  5      OPC=callq_label     
.L_ff585:                       #        0xff591  0      OPC=<label>         
  movq 0x29b8f4(%rip), %rax     #  205   0xff591  7      OPC=movq_r64_m64    
  xorl %r15d, %r15d             #  206   0xff598  3      OPC=xorl_r32_r32    
  movl $0x69, (%rax)            #  207   0xff59b  6      OPC=movl_m32_imm32  
  nop                           #  208   0xff5a1  1      OPC=nop             
  jmpq .L_ff535                 #  209   0xff5a2  2      OPC=jmpq_label      
.L_ff598:                       #        0xff5a4  0      OPC=<label>         
  addq $0x38, %rsp              #  210   0xff5a4  4      OPC=addq_r64_imm8   
  xorl %eax, %eax               #  211   0xff5a8  2      OPC=xorl_r32_r32    
  popq %rbx                     #  212   0xff5aa  1      OPC=popq_r64_1      
  popq %rbp                     #  213   0xff5ab  1      OPC=popq_r64_1      
  popq %r12                     #  214   0xff5ac  2      OPC=popq_r64_1      
  popq %r13                     #  215   0xff5ae  2      OPC=popq_r64_1      
  popq %r14                     #  216   0xff5b0  2      OPC=popq_r64_1      
  popq %r15                     #  217   0xff5b2  2      OPC=popq_r64_1      
  retq                          #  218   0xff5b4  1      OPC=retq            
.L_ff5a9:                       #        0xff5b5  0      OPC=<label>         
  movl $0x10, %edi              #  219   0xff5b5  5      OPC=movl_r32_imm32  
  callq .memalign_plt           #  220   0xff5ba  5      OPC=callq_label     
  testq %rax, %rax              #  221   0xff5bf  3      OPC=testq_r64_r64   
  movq %rax, %r15               #  222   0xff5c2  3      OPC=movq_r64_r64    
  je .L_ff585                   #  223   0xff5c5  2      OPC=je_label        
  xorl %eax, %eax               #  224   0xff5c7  2      OPC=xorl_r32_r32    
  jmpq .L_ff524                 #  225   0xff5c9  5      OPC=jmpq_label_1    
.L_ff5c2:                       #        0xff5ce  0      OPC=<label>         
  leaq 0x24(%rbp), %rdi         #  226   0xff5ce  4      OPC=leaq_r64_m16    
  leaq -0x4(%r8), %rsi          #  227   0xff5d2  4      OPC=leaq_r64_m16    
  jmpq .L_ff55f                 #  228   0xff5d6  2      OPC=jmpq_label      
  nop                           #  229   0xff5d8  1      OPC=nop             
  nop                           #  230   0xff5d9  1      OPC=nop             
  nop                           #  231   0xff5da  1      OPC=nop             
  nop                           #  232   0xff5db  1      OPC=nop             
                                                                             
.size if_nameindex, .-if_nameindex
