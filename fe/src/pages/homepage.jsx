import React from 'react'
import Card from './card'
import { div } from 'three/examples/jsm/nodes/Nodes.js'

export default function Homepage({activepagechanger , profiledata, userid ,toppicks, userchoice , viewproducts}) {
  return (
    <div className='text-white flex justify-start gap-6 w-screen h-screen'>
        <div className=' w-[25%] h-screen border-r border-r-zinc-800 bg-stone-950'>
            <div className='text-center text-2xl font-bold tracking-[10px] py-6  bg-gradient-to-b from-black via-pink-800 to-black bg-clip-text text-transparent'>SEPHORA</div>

            <div className='mt-12 flex justify-center'>
                <div className=' px-[25px] text-[50px] rounded-full bg-zinc-900 border-2 border-zinc-700 text-center '>P</div>
            </div>
            <div className='text-center text-stone-700'>User ID</div>
            <div className='text-center text-3xl font-semibold text-zinc-300'>{profiledata?.user_id}</div>
            
            <div className='flex justify-between px-6 py-3 mt-12 mx-12 bg-neutral-900 rounded-3xl border border-neutral-700 text-xl font-semibold'>
                <div>Total Reactions</div>
                <div>{profiledata?.love_count}</div>
            </div>
            <div className='flex justify-between px-6 py-3 mt-4 mx-12 bg-neutral-900 rounded-3xl border border-neutral-700 text-xl font-semibold'>
                <div>Total Feedbacks</div>
                <div>{profiledata?.total_reviews}</div>
            </div>

            <div className='mx-12 mt-5 '>

                <div className='text-stone-500 pb-3 font-semibold '>PRODUCT PREFERENCES</div>
                <div className='flex flex-wrap gap-2 justify-start'>
                {[profiledata?.top_categories?.[0], profiledata?.top_brands?.[0]]
                    .filter(Boolean)   // removes undefined/null
                    .map((k, index) => (
                        <div
                        key={index}
                        className="px-6 py-1 text-white bg-neutral-900 rounded-full border border-neutral-700 text-sm"
                        >
                        {k}
                        </div>
                    ))}
                </div>
            </div>

            <div className='mx-12 mt-16'>
                <div className='text-center py-3 font-semibold bg-rose-700 rounded-xl mt-12 cursor-pointer hover:shadow-sm hover:shadow-rose-600 duration-300' onClick={()=>activepagechanger(1)}>Sign Out</div>
            </div>

            <div>
                
            </div>
        </div>

        
    
      <div className='py-6'>
        <div className='font-bold text-xl'>Based On User Preferences</div>
        <div className='flex gap-2 mt-6 overflow-x-auto w-[70vw] h-[320px] scrollbar-hide '>
            {userchoice?.map((item,index) => (
                <div
                    key={item.product_id}
                    className="w-[270px] h-[300px]"
                    onClick={() => viewproducts(3,index,1,item.product_id)}
                >
                    <Card
                    productId={item.product_id}
                    productName={item.product_name}
                    category = {item.category}
                    price = {item.price}
                    brand = {item.brand}
                    love_count = {item.love_count}
                    rating = {item.rating}
                    size={item.size}
                    product_id = {item.product_id}
                    />
                </div>
            ))}
            
        </div>


        <div className='font-bold mt-6 text-xl'>Top Picked Items</div>
        <div className='flex gap-2 mt-6 overflow-x-auto w-[70vw] h-[320px] scrollbar-hide '>
            {toppicks?.map((item,index) => (
                <div
                    key={item.product_id}
                    className="w-[270px] h-[300px]"
                    onClick={() => viewproducts(3, index,2,item.product_id)}
                >
                    <Card
                    productId={item.product_id}
                    productName={item.product_name}
                    category = {item.primary_category}
                    price = {item.price}
                    brand = {item.brand_name}
                    love_count = {item.love_count}
                    rating = {item.avg_rating}
                    size={item.variation_value}
                    product_id = {item.product_id}
                    />
                </div>
            ))}
            
        </div>
      </div>
    </div>
  )
}
